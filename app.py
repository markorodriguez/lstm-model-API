from flask import Flask, request

from src.service.predict import predict_text

app = Flask(__name__)

@app.route('/')
def index():
    return 'Main Route'

@app.post('/api/trained-model')
def trained_model():
    body = request.get_json()
    predict_text(body['text'])
      
