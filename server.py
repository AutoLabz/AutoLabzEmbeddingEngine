import jwt, datetime, os, time
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from pinecone.grpc import PineconeGRPC as Pinecone

# Load environment variables
load_dotenv()
server = Flask(__name__)



# Check if the Flask server is starting
print("Starting Flask server...")

# Initialize SentenceTransformer Model & PineCone Client
print("Initializing SentenceTransformer model and PineCone client...")


model = SentenceTransformer('all-MiniLM-L6-v2')
bm25 = BM25Encoder()

pineconeObj = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Create serverless index with dimension 384
index_name = "car-search-hybrid"
if index_name not in pineconeObj.list_indexes().names():
    print("Index not found, creating index...")
    pineconeObj.create_index(
        name=index_name,
        dimension=384,  
        metric="dotproduct",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    while not pineconeObj.describe_index(index_name).status['ready']:
        print("Waiting for index to be ready...")
        time.sleep(1)

print("Connecting to the index...")
index = pineconeObj.Index(index_name)
index_stats = index.describe_index_stats()
print("Index connected and ready.")

# Create sparse-dense vector embeddings for an array of car descriptions
@server.post('/embedDescriptions')
def embed_descriptions():
    print("HIT PYTHON SERVICE!")
    car_data = request.json.get('car_data', [])
    print("Received car data: ", car_data)

    if not car_data:
        return jsonify({'error': 'No car data provided'}), 400

    # These are dense vectors
    dense_embeddings = model.encode(car_data)
    print("The shape of the dense vector embeddings are ", dense_embeddings.shape)

    # Dense Embeddings are created at this point so semantic part is half done. Now to create sparse vectors.
    
    bm25.fit(car_data)
    sparse_vectors = bm25.encode_documents(car_data)
    print("Sparse vectors created.")

    # Prepare vectors for upsert and map vector IDs to car objects
    vectors = []
    vector_ids = []
    for i, (dense, sparse) in enumerate(zip(dense_embeddings, sparse_vectors)):
        vector_id = f'vec{i}'
        vector = {
            'id': vector_id,
            'values': dense.tolist(),
            'metadata': {'description': car_data[i]},
            'sparse_values': {
                'indices': sparse['indices'],
                'values': sparse['values']
            }
        }
        vectors.append(vector)
        vector_ids.append(vector_id)

    # Upsert vectors into Pinecone index
    upsert_response = index.upsert(
        vectors=vectors,
        namespace='cars-testing-namespace'
    )

    # Return the vector IDs
    return jsonify({"vector_ids": vector_ids})



# Create a sparse-dense vector embedding of a single car description
@server.post('/embedDescription')
def embed_description():
    print("Hit /embedDescriptions endpoint!")
    car_data = request.json.get('car_data', '')
    print("Received car data: ", car_data)

    if not car_data:
        return jsonify({'error': 'No car data provided'}), 400

    # These are dense vectors
    dense_embeddings = model.encode(car_data)
    print("The shape of the dense vector embeddings are ", dense_embeddings.shape)

    # Dense Embeddings are created at this point so semantic part is half done. Now to create sparse vectors.
    
    bm25.fit(car_data)
    sparse_vectors = bm25.encode_documents(car_data)
    print("Sparse vectors created.")

    # Prepare vectors for upsert and map vector IDs to car objects
    vectors = []
    vector_ids = []
    for i, (dense, sparse) in enumerate(zip(dense_embeddings, sparse_vectors)):
        vector_id = f'vec{i}'
        vector = {
            'id': vector_id,
            'values': dense.tolist(),
            'metadata': {'description': car_data[i]},
            'sparse_values': {
                'indices': sparse['indices'],
                'values': sparse['values']
            }
        }
        vectors.append(vector)
        vector_ids.append(vector_id)

    # Upsert vectors into Pinecone index
    upsert_response = index.upsert(
        vectors=vectors,
        namespace='cars-testing-namespace'
    )

    # Return the vector IDs
    return jsonify({"vector_ids": vector_ids})


# Define the search endpoint
@server.get('/search')
def search():
    query_text = request.args.get('query', '')
    if not query_text:
        return jsonify({'error': 'No query provided'}), 400

    # Create dense vector
    dense_vector = model.encode([query_text])[0]
    print("Dense vector created!")

    # Create sparse vector
    bm25 = BM25Encoder()
    bm25.fit([query_text])
    sparse_vector = bm25.encode_queries([query_text])[0]
    print("Sparse vector created!")

    # Query the index
    query_response = index.query(
        namespace="cars-testing-namespace",
        top_k=10,
        vector=dense_vector,
        sparse_vector=sparse_vector
    )
    print("Query response: ", query_response)

    # Extract the vector IDs from the query response
    vector_ids = [match['id'] for match in query_response['matches']]

    return jsonify({"vector_ids": vector_ids})

# Endpoint to get namespaces and their stats
@server.get('/namespaces')
def get_namespaces():
    #index_stats = index.describe_index_stats()
    namespaces = index_stats['namespaces']
    
    namespace_list = []
    for namespace, stats in namespaces.items():
        namespace_list.append({"namespace": namespace, "vector_count": stats['vector_count']})
    
    return jsonify({"namespaces": namespace_list})


# Run the Flask server
if __name__ == '__main__':
    print("Running Flask server...")
    server.run(host='0.0.0.0', port=5005)
