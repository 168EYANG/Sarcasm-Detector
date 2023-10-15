import io
import json
import torch
from torch import nn
import numpy as np
from flask import Flask, jsonify, request


app = Flask(__name__)
model = torch.load('./SentimentAnalysis/trained_model.pt')
model.eval()

#FIXME
def get_prediction():
    # tokenize
    # prediction = model.forward(x)
    # return the prediction
    return


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # get text from POST Request
        sent_pred = get_prediction() # Send text to get_prediction
        return jsonify({'sentiment': sent_pred})


if __name__ == '__main__':
    app.run()