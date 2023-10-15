import io
import json
import torch
from torch import nn
import numpy as np
from flask import Flask, jsonify, request
import model as sm
import re
import html
from nltk.tokenize import word_tokenize


app = Flask(__name__)
model = sm.SarcasmModel(300)
model.load_state_dict(torch.load('trained_model_newest.pt'))
model.eval()

def spec_add_spaces(t: str) -> str:
    "Add spaces around / and # in `t`. \n"
    return re.sub(r"([/#\n])", r" \1 ", t)

def rm_useless_spaces(t: str) -> str:
    "Remove multiple spaces in `t`."
    return re.sub(" {2,}", " ", t)

def replace_multi_newline(t: str) -> str:
    return re.sub(r"(\n(\s)*){2,}", "\n", t)

def fix_html(x: str) -> str:
    "List of replacements from html strings in `x`."
    re1 = re.compile(r"  +")
    x = (
        x.replace("#39;", "'")
        .replace("amp;", "&")
        .replace("#146;", "'")
        .replace("nbsp;", " ")
        .replace("#36;", "$")
        .replace("\\n", "\n")
        .replace("quot;", "'")
        .replace("<br />", "\n")
        .replace('\\"', '"')
        .replace(" @.@ ", ".")
        .replace(" @-@ ", "-")
        .replace(" @,@ ", ",")
        .replace("\\", " \\ ")
    )
    return re1.sub(" ", html.unescape(x))

def clean_text(input_text):
    text = fix_html(input_text)
    text = replace_multi_newline(text)
    text = spec_add_spaces(text)
    text = rm_useless_spaces(text)
    text = text.strip()
    return text

vocab, embeddings = [],[]
with open('glove.6B.300d.txt','rt', encoding="utf8") as fi:
    full_content = fi.read().strip().split('\n')
for i in range(len(full_content)):
    i_word = full_content[i].split(' ')[0]
    i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
    vocab.append(i_word)
    embeddings.append(i_embeddings)
vocab_npa = np.array(vocab)
embs_npa = np.array(embeddings)
vocab_npa = np.insert(vocab_npa, 0, '<pad>')
vocab_npa = np.insert(vocab_npa, 1, '<unk>')
print(vocab_npa[:10])
vocab_npa_dict = dict()
for i in range(len(vocab_npa)):
    vocab_npa_dict[vocab_npa[i]] = i

pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.

#insert embeddings for pad and unk tokens at top of embs_npa.
embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))
print(embs_npa[:10])

PADDING = 0
UNKNOWN_WORD = 1
MAX_COMMENT_SIZE = 50

output_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

def get_prediction(text):
    # tokenize
    text = clean_text(text)
    text = text.lower()
    text = word_tokenize(text)

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
    prediction = model.forward(text)
    return np.argmax(torch.Tensor.detach(torch.Tensor.cpu(prediction)).numpy())


@app.route('/predict/<jsdata>')
def predict(jsdata):
    sent_pred = get_prediction(jsdata)
    return jsonify({'sentiment': output_map[sent_pred]})


if __name__ == '__main__':
    app.run()