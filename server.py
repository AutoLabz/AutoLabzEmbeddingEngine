import jwt, datetime, os
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from transformers import BertTokenizer
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder


load_dotenv()
server = Flask(__name__)


model = SentenceTransformer('all-MiniLM-L6-v2')
pineconeObj = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

@server.route('/embed', methods=['POST'])
def embed():
    print("HIT PYTHON SERVICE!")
    car_data = request.json.get('car_data', [])
    print(car_data)

    embeddings = []
    embeddings = model.encode(car_data)

    print("The shape of the embeddings are ", embeddings.shape)

    # Dense Embeddings are created at this point so sematic part is half done. Now to create sparse vectors. 

    bm25 = BM25Encoder()
    bm25.fit(car_data)

    sparse_vectors = bm25.encode_documents(car_data)
    for vector in sparse_vectors:
        print(vector)
  

    if not car_data:
        return jsonify({'error': 'No car data provided'}), 400
    

    #return jsonify({'embeddings': embeddings})
    return None




@server.route('/search', methods=['GET'])
def search():


if __name__ == '__main__':
    server.run(host='0.0.0.0', port=5005)




'''
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    all_tokens = [tokenizer.tokenize(sentence.lower()) for sentence in car_data]
    print(all_tokens[0])

'''