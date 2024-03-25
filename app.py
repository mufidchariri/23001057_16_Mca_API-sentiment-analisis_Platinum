from flask import Flask, jsonify, request
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from


import pickle, re
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer

class UploadFilecsv(Flask):
    json_provider_class = LazyJSONEncoder
app = UploadFilecsv(__name__)
# app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
    'title' : LazyString(lambda: 'API Documentation for Deep Learning'),
    'version' : LazyString(lambda: '1.0.0'),
    'description' : LazyString(lambda: 'Dokumentasi API untuk Deep Learning'),
    },
    host = LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template, config=swagger_config)

max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

sentiment = ['negative', 'neutral', 'positive']

file_lstm = open('resources_of_lstm/x_pad_sequences3.pickle', 'rb')
feature_lstm = pickle.load(file_lstm)
file_lstm.close()

#load model lstm
model_lstm = load_model('notebook/model_lstm_platinum.h5')
print(model_lstm.summary())

#load model neural network
count_vect = CountVectorizer()
count_vect = pickle.load(open('notebook/feature_platinum_nn.p', 'rb'))
loaded_model = pickle.load(open('notebook/model_platinum_nn.p', 'rb'))

# def cleansing
def cleansing(sent):
    string = sent.lower()
    string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
    return string

@swag_from('docs/neural_network.yml', methods=['POST'])
@app.route('/Analisis sentimen Neural Network', methods=['POST'])
def nn():
    original_text = request.form.get('text')
    text = count_vect.transform([cleansing(original_text)])
    result = loaded_model.predict(text)[0]

    json_response = {
        'status_code' : 200,
        'description' : 'Hasil Analisis Sentimen dengan Neural Network',
        'data' : {
            'text' : original_text,
            'sentiment' : result
        },
    }
    response_data = jsonify(json_response)
    return response_data

@swag_from('docs/upload2.yml', methods=['POST'])
@app.route('/Analisis Sentiment Data Tweet menggunakan Neural Network', methods=['POST'])
def upload_nn():
    file = request.files.get('file')
    
    csv_data = pd.read_csv(file, encoding='ISO-8859-1')
    tweet = csv_data['Tweet']

    def cleantext(sent):
        string = sent.lower()
        string = re.sub(r'[^a-z0-9A-Z]', ' ', string)
        string = re.sub('user', '', string)
        string = re.sub('\t', '', string)
        return string
    
    tweet_clean = tweet.apply(cleantext)
    prediksi_data = count_vect.transform(tweet_clean.tolist())
    result = loaded_model.predict(prediksi_data)

    json_response = {
        'status_code' : 200,
        'description' : 'Hasil Analisis Sentimen dengan Neural Network',
        'data' : []
    }
    for i in range(len(tweet_clean)):
        json_response['data'].append({
            'text' : tweet_clean[i],
            'sentiment' : result[i].tolist()
        })
    response_data = jsonify(json_response)
    return response_data

@swag_from('docs/lstm.yml', methods=['POST'])
@app.route('/Analisis sentimen LSTM', methods=['POST'])
def lstm():
    original_text = request.form.get('text')
    text = [cleansing(original_text)]
    
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_lstm.shape[1])

    prediction = model_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]
    json_response = {
        'status_code' : 200,
        'description' : 'Hasil Analisis Sentimen dengan LSTM',
        'data' : {
            'text' : original_text,
            'sentiment' : get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

@swag_from('docs/upload1.yml', methods=['POST'])
@app.route('/Analisis Sentiment Data Tweet menggunakan LSTM', methods=['POST'])
def upload_lstm():

    file = request.files.get('file')
    
    csv_data = pd.read_csv(file, encoding='ISO-8859-1')
    tweet = csv_data['Tweet']

    def cleantext(sent):
        string = sent.lower()
        string = re.sub(r'[^a-z0-9A-Z]', ' ', string)
        string = re.sub('user', '', string)
        string = re.sub('\t', '', string)
        return string
    
    tweet_clean = tweet.apply(cleantext)
        
    feature = tokenizer.texts_to_sequences(tweet_clean)
    feature = pad_sequences(feature, maxlen=feature_lstm.shape[1])

    prediction = model_lstm.predict(feature)
    tweet_sentiments = []
    for idx, pred in enumerate(prediction):
        sentiment_idx = np.argmax(pred)
        sentiment_label = sentiment[sentiment_idx]
        tweet_sentiments.append({'text': tweet_clean.iloc[idx], 'sentiment': sentiment_label})
    
    json_response = {
        'status_code' : 200,
        'description' : 'Hasil Analisis Sentimen Data Tweet dengan LSTM',
        'data' : tweet_sentiments,
    }

    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run()