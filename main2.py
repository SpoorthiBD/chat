import json 
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Dense, Embedding, GlobalAveragePooling1D
# from keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder

import colorama 
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

# app = Flask(__name__)
# CORS(app)

# @app.route('/')
# def welcome():
#     return ('Welcome to the Chatbot!')

# app = Flask(__name__)

# @app.route('/')
# def welcome():
#     return jsonify({'message': 'Welcome to the Chatbot Backend Server!'})


with open("intents.json") as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
        
num_classes = len(labels)

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

model.summary()

epochs = 500
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)


model.save("chat_model")

import pickle

# to save the fitted tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# to save the fitted label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    finalResponse = ''
    # return jsonify({'response':'samath'})
    # load trained model
    model = keras.models.load_model('chat_model')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    while True:
        # print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        # inp = input()
        inp = user_message
        if inp.lower() == "quit":
            finalResponse = "Good Bye ! Samatha"
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                if i['showAll']:
                    tempList = ''
                    for k in i['responses']:
                        # tempList += '<a href="'+k["link"]+'" target="_blank">'+k["text"]+'</a></br>'
                        if isinstance(k, str) == False:
                            # tempList += '<a href="'+k["link"]+'" target="_blank">'+k["text"]+'</a></br>'
                            tempList += "<a href='javascript:;' onclick='return displaySelectedButton(\""+k['link']+"\")'>"+k['text']+"</a></br>"
                        else:
                            tempList += k
                    finalResponse = tempList
                else:
                    finalResponse = np.random.choice(i["responses"])
                # finalResponse = np.random.choice(i["responses"])
                # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))
                return jsonify({'response': finalResponse})
        
        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))

# print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
# chat()


# @app.route('/favicon.ico')
# def favicon():
#     return '', 404  



if __name__ == '__main__':
    app.run(debug=True)
