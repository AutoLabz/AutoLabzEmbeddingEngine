import jwt, datetime, os
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
server = Flask(__name__)


model = SentenceTransformer('all-MiniLM-L6-v2')

@server.route('/embed', methods=['POST'])
def embed():
    print("HIT PYTHON SERVICE!")
    car_data = request.json.get('car_data', [])
    if not car_data:
        return jsonify({'error': 'No car data provided'}), 400
    
    # Create a list of embeddings for each car's attributes
    embeddings = []
    for car in car_data:
        car_embeddings = {}
        for key, value in car.items():
            # Ensure the value is a string for the embedding model
            value_str = str(value)
            embedding = model.encode([value_str])[0]
            car_embeddings[key] = embedding.tolist()
        embeddings.append(car_embeddings)
    #print(embeddings)

    return jsonify({'embeddings': embeddings})

if __name__ == '__main__':
    server.run(host='0.0.0.0', port=5005)