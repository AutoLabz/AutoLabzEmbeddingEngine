import jwt, datetime, os, time
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from transformers import BertTokenizer
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
pineconeObj = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Create serverless index
print("Creating serverless index if it doesn't exist...")
index_name = "hybrid-car-search"
if index_name not in pineconeObj.list_indexes().names():
    print("Index not found, creating index...")
    pineconeObj.create_index(
        name=index_name,
        dimension=1536,
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
index.describe_index_stats()
print("Index connected and ready.")

# Define the embedding endpoint
@server.post('/embed')
def embed():
    print("HIT PYTHON SERVICE!")
    car_data = request.json.get('car_data', [])
    print("Received car data: ", car_data)

    if not car_data:
        return jsonify({'error': 'No car data provided'}), 400

    # These are dense vectors
    dense_embeddings = model.encode(car_data)
    print("The shape of the embeddings are ", dense_embeddings.shape)

    # Dense Embeddings are created at this point so semantic part is half done. Now to create sparse vectors.
    bm25 = BM25Encoder()
    bm25.fit(car_data)
    sparse_vectors = bm25.encode_documents(car_data)
    print("Sparse vectors created.")

    return jsonify({'embeddings': dense_embeddings.tolist()})

# Define the search endpoint
@server.get('/search')
def search():
    return jsonify({'message': 'Search endpoint hit'})

# Run the Flask server
if __name__ == '__main__':
    print("Running Flask server...")
    server.run(host='0.0.0.0', port=5005)
