import io
import json
import torch
from torch import nn
import numpy as np
from flask import Flask, jsonify, request
import model as sm
import textClean as tc
from nltk.tokenize import word_tokenize
import generate_dicts as gd

app = Flask(__name__)



embs_npa = gd.getEmbs()
vocab_npa_dict = gd.getVoc()

model = sm.SarcasmModel(300, embs_npa)
model.load_state_dict(torch.load('trained_model_newest.pt'))
model.eval()

PADDING = 0
UNKNOWN_WORD = 1
MAX_COMMENT_SIZE = 50

output_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

def get_prediction(text):
    # tokenize
    text = tc.clean_text(text)
    text = text.lower()
    text = word_tokenize(text)

    # numberify
    # numberify
    new_text = []
    for counter in range(min(len(text), MAX_COMMENT_SIZE)):
        word = text[counter]
        if word not in vocab_npa_dict.keys():
            new_text.append(UNKNOWN_WORD)
        else:
            new_text.append(vocab_npa_dict[word])
    while len(new_text) < MAX_COMMENT_SIZE:
        new_text.append(PADDING)
    
    # predict and return it
    prediction = model.forward(new_text)
    return np.argmax(torch.Tensor.detach(torch.Tensor.cpu(prediction)).numpy())


@app.route('/predict/<jsdata>')
def predict(jsdata):
    sent_pred = get_prediction(jsdata)
    return jsonify({'sentiment': output_map[sent_pred]})


if __name__ == '__main__':
    app.run()