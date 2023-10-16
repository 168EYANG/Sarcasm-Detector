import numpy as np
import pickle
import os.path

def generateDicts():
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

    np.save('emb_dict.npy', embs_npa)
    with open('voc_dict.pkl', 'wb') as fp:
        pickle.dump(vocab_npa_dict, fp)
    
def getEmbs():
    if not (os.path.isfile('emb_dict.npy') and os.path.isfile('voc_dict.pkl')):
        generateDicts()
    embs_npa = np.load('emb_dict.npy')
    return embs_npa
    

def getVoc():
    if not (os.path.isfile('emb_dict.npy') and os.path.isfile('voc_dict.pkl')):
        generateDicts()
    with open('voc_dict.pkl', 'rb') as fp:
        vocab_npa_dict = pickle.load(fp)
    return vocab_npa_dict