import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
# from nltk import pos_tag
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from collections import defaultdict
# from nltk.corpus import wordnet as wn
import re
import html
import ast
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
BATCH_SIZE = 32
TOTAL_EPOCHS = 30

# text / sentiment / negative-neutral-positive
data_1 = pd.read_csv("train_1.csv", on_bad_lines='skip', encoding='unicode_escape')
data_1 = data_1[data_1['text'].notna()]
data_1 = data_1[data_1['sentiment'].isin(['negative', 'neutral', 'positive'])]
data_1_specific = data_1.loc[:, ['text', 'sentiment']]

# text / sentiment / negative-neutral-positive
data_2 = pd.read_csv("train_2.csv", on_bad_lines='skip', encoding='unicode_escape')
data_2 = data_2[data_2['text'].notna()]
data_2 = data_2[data_2['sentiment'].isin(['negative', 'neutral', 'positive'])]
data_2_specific = data_2.loc[:, ['text', 'sentiment']]

# text / sentiment / Negative-Neutral-Positive
data_3 = pd.read_csv("train_3.csv", on_bad_lines='skip', encoding='unicode_escape')
data_3 = data_3[data_3['text'].notna()]
data_3 = data_3[data_3['sentiment'].isin(['negative', 'neutral', 'positive'])]
data_3_specific = data_3.loc[:, ['text', 'sentiment']]

data = pd.concat([data_1_specific, data_2_specific, data_3_specific], ignore_index=True)
data_len = len(data)
sentiment_map = {'negative':[1,0,0], 'neutral':[0,1,0], 'positive':[0,0,1]}

for i in tqdm(range(data_len)):
    data.at[i, 'text'] = clean_text(data.iloc[i]['text'])
    data.at[i, 'text'] = data.at[i, 'text'].lower()
    data.at[i, 'text'] = word_tokenize(data.at[i, 'text'])
    data.at[i, 'sentiment'] = sentiment_map[data.at[i, 'sentiment']]
data.to_json('train-balanced-sarcasm-processed.json')

# data = pd.read_json("train-balanced-sarcasm-processed.json")
data_len = len(data)

# create dictionary of words, and word-to-id dictionary
# word_dict = dict()
# for i in tqdm(range(data_len)):
#     for word in data.iloc[i]["comment"]:
#         if word not in word_dict:
#             word_dict[word] = 0
#         word_dict[word] += 1
# print(len(word_dict))

# word_to_id = {"": PADDING, "UNK": UNKNOWN_WORD}
# running_id = 2
# for key in word_dict:
#     if word_dict[key] > 10:
#         word_to_id[key] = running_id
#         running_id += 1
# print(len(word_to_id))

# convert all comment arrays to arrays of numbers
for i in tqdm(range(data_len)):
    new_comment = []
    old_comment = data.iloc[i]["text"]
    for counter in range(min(len(old_comment), MAX_COMMENT_SIZE)):
        word = old_comment[counter]
        if word not in vocab_npa_dict.keys():
            new_comment.append(UNKNOWN_WORD)
        else:
            new_comment.append(vocab_npa_dict[word])
    while len(new_comment) < MAX_COMMENT_SIZE:
        new_comment.append(PADDING)
    data.at[i, "text"] = new_comment
data.to_json('train-balanced-sarcasm-numbered.json')

embedding_dims = 300

data = pd.read_json("train-balanced-sarcasm-numbered.json")
data_len = len(data)

# oops i forgot the vocabulary max size
# max_vocab = 27080
max_vocab = 0
for i in tqdm(range(data_len)):
    for word in data.iloc[i]["text"]:
        if word == 0:
            break
        max_vocab = max(max_vocab, word)
print(max_vocab)

class SarcasmDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        tokenized_word_tensor = torch.tensor(self.df.iloc[index]["text"], dtype=torch.long)
        curr_label = self.df.iloc[index]["sentiment"]
        return tokenized_word_tensor, curr_label

class SarcasmModel(nn.Module):
    def __init__(self, embedding_dim) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float()).to(device)

        assert self.embedding.weight.shape == embs_npa.shape
        print(self.embedding.weight.shape)
        # self.conv = nn.Conv1d(embedding_dim, 32, kernel_size=3, padding="same", device=device)
        self.lstm = nn.LSTM(embedding_dim, 196, bidirectional=True, batch_first=True, device=device, dropout=0.5, num_layers=2)
        self.linear_1 = nn.Linear(196*2, 3, device=device)
        #self.linear_2 = nn.Linear(64, 32, device=device)
        #self.linear_3 = nn.Linear(32, 3, device=device)
    def forward(self, x):
        embed_full = self.embedding(x)
        # conved = nn.functional.relu(self.conv(embed_full))
        # conved, _ = conved.max(dim = -1)
        lstm_output, _ = self.lstm(embed_full)
        max_pooled = torch.max(lstm_output, axis=1).values.view(BATCH_SIZE, 196*2)
        #hidden_1 = nn.functional.dropout1d(nn.functional.relu(self.linear_1(max_pooled)), 0.2)
        #hidden_2 = nn.functional.dropout1d(nn.functional.relu(self.linear_2(hidden_1)), 0.2)
        output = nn.functional.sigmoid(self.linear_1(max_pooled))
        # output = nn.functional.sigmoid(self.linear_1(conved))
        return output.view(-1, 3)

def collate(batch, padding_value = PADDING):
    padded_tokens = []
    y_labels = []
    for tensor_tuple in batch:
        padded_tokens.append(tensor_tuple[0])
        y_labels.append(tensor_tuple[1])
    padded_tokens = torch.nn.utils.rnn.pad_sequence(padded_tokens, batch_first=True, padding_value=padding_value).to(device)
    y_labels = torch.tensor(y_labels, dtype=torch.float).to(device)
    return padded_tokens, y_labels

data = data.sample(frac=1)
thresh = int(data_len * 0.8)
train_df = data.iloc[:thresh]
train_dataset = SarcasmDataset(train_df)
test_df = data.iloc[thresh:]
test_dataset = SarcasmDataset(test_df)

train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(train_dataset), collate_fn=collate)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(test_dataset), collate_fn=collate)

model = SarcasmModel(embedding_dims)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

loss_graph = pd.DataFrame(columns=['id', 'total loss'])
acc_graph = pd.DataFrame(columns=['id', 'train acc', 'val acc'])

for epoch in range(TOTAL_EPOCHS):

    counter = 0
    pbar = tqdm(train_iterator)
    total_loss = 0
    model.train()
    train_correct = 0
    train_total = 0
    for x, true in pbar:
        if len(x) != BATCH_SIZE:
            continue
        model.zero_grad()
        pred = model.forward(x)
        for i in range(len(true)):
            if np.argmax(torch.Tensor.detach(torch.Tensor.cpu(true[i])).numpy()) == np.argmax(torch.Tensor.detach(torch.Tensor.cpu(pred[i])).numpy()):
                train_correct += 1
        train_total += len(true)
        loss = loss_fn(pred, true)
        loss.backward()
        optimizer.step()
        counter += 1
        total_loss += loss.item()
    train_acc = train_correct / train_total

    valbar = tqdm(test_iterator)
    val_correct = 0
    val_total = 0
    for x, true in valbar:
        if len(x) != BATCH_SIZE:
            continue
        pred = torch.round(model.forward(x))
        for i in range(len(true)):
            if np.argmax(torch.Tensor.detach(torch.Tensor.cpu(true[i])).numpy()) == np.argmax(torch.Tensor.detach(torch.Tensor.cpu(pred[i])).numpy()):
                val_correct += 1
        val_total += len(true)
    val_acc = val_correct / val_total
    
    loss_graph.loc[len(loss_graph.index)] = [epoch, total_loss]
    acc_graph.loc[len(acc_graph.index)] = [epoch, train_acc, val_acc]
    
    print("----- RESULTS OF EPOCH ", str(epoch), " -----")
    print("total loss: ", str(total_loss))
    print("train acc: ", str(train_acc))
    print("val acc: ", str(val_acc))

    torch.save(model, "trained_model.pt")

loss_graph.plot(x="id", y="total loss")
acc_graph.plot(x="id", y="train acc")
acc_graph.plot(x="id", y="val acc")

# ##hyper parameters
# batch_size = 64
# embedding_dims = 300 #Length of the token vectors
# filters = 250 #number of filters in your Convnet
# kernel_size = 3 # a window size of 3 tokens
# hidden_dims = 250 #number of neurons at the normal feedforward NN
# epochs = 1

# def unweirdify(array):
#     new_array = []
#     for i in range(len(array)):
#         new_array.extend(array[i])
#     return np.asarray(new_array).reshape((-1, MAX_COMMENT_SIZE))

# data = pd.read_json("train-balanced-sarcasm-numbered.json")
# data = data[data["label"].notna()]
# data_len = len(data)

# data = data.sample(frac=1)
# thresh = int(data_len * 0.9)
# x_train = np.asarray(unweirdify(data.iloc[:thresh]["comment"].to_numpy())).astype('float32')
# y_train = data.iloc[:thresh]["label"].to_numpy().astype('float32')
# for i in y_train:
#     if i != 0.0 and i != 1.0:
#         print(i)
# x_test = np.asarray(unweirdify(data.iloc[thresh:]["comment"].to_numpy())).astype('float32')
# y_test = data.iloc[thresh:]["label"].to_numpy().astype('float32')
# print(x_train.shape)

# # oops i forgot the vocabulary max size
# max_vocab = 27080
# # max_vocab = 0
# # for i in tqdm(range(data_len)):
# #     for word in data.iloc[i]["comment"]:
# #         if word == 0:
# #             break
# #         max_vocab = max(max_vocab, word)
# # print(max_vocab)

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Embedding(max_vocab+1, embedding_dims, input_length=MAX_COMMENT_SIZE))
# model.add(tf.keras.layers.Conv1D(filters,kernel_size,padding = 'valid' , activation = 'relu',strides = 1 , input_shape = (MAX_COMMENT_SIZE,embedding_dims)))
# model.add(tf.keras.layers.GlobalMaxPooling1D())
# model.add(tf.keras.layers.Dense(hidden_dims))
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.Dense(1))
# model.add(tf.keras.layers.Activation('sigmoid'))
# model.compile(loss = 'binary_crossentropy',optimizer = 'adam', metrics = ['accuracy'])
# K.set_value(model.optimizer.learning_rate, 0.001)
# model.fit(x_train,y_train,batch_size = batch_size,epochs = epochs , validation_data = (x_test,y_test))
# model.save('trained_model.keras')

# PADDING = 0
# UNKNOWN_WORD = 1
# MAX_COMMENT_SIZE = 400

# data = pd.read_json("train-balanced-sarcasm-processed.json")
# data_len = len(data)

# # create dictionary of words, and word-to-id dictionary
# word_dict = dict()
# for i in tqdm(range(data_len)):
#     for word in data.iloc[i]["comment"]:
#         if word not in word_dict:
#             word_dict[word] = 0
#         word_dict[word] += 1
# print(len(word_dict))

# word_to_id = {"": PADDING, "UNK": UNKNOWN_WORD}
# running_id = 2
# for key in word_dict:
#     if word_dict[key] > 1:
#         word_to_id[key] = running_id
#         running_id += 1
# print(len(word_to_id))

# # convert all comment arrays to arrays of numbers
# for i in tqdm(range(data_len)):
#     new_comment = []
#     old_comment = data.iloc[i]["comment"]
#     for counter in range(min(len(old_comment), MAX_COMMENT_SIZE)):
#         word = old_comment[counter]
#         if word not in word_to_id.keys():
#             new_comment.append(UNKNOWN_WORD)
#         else:
#             new_comment.append(word_to_id[word])
#     while len(new_comment) < 400:
#         new_comment.append(PADDING)
#     data.at[i, "comment"] = new_comment
# data.to_json('train-balanced-sarcasm-numbered.json')

# data = pd.read_csv("train-balanced-sarcasm.csv", on_bad_lines='skip')
# data = data[data['comment'].notna()]
# data_len = len(data)

# for i in tqdm(range(data_len)):
#     data.at[i, 'comment'] = clean_text(data.iloc[i]['comment'])
#     data.at[i, 'comment'] = data.at[i, 'comment'].lower()
#     data.at[i, 'comment'] = word_tokenize(data.at[i, 'comment'])
# data.to_json('train-balanced-sarcasm-processed.json')

# # Step - a : Remove blank rows if any.
# # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
# # data['comment'] = [entry.lower() for entry in data['comment']]
# # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
# # data['comment']= [word_tokenize(entry) for entry in data['comment']]
# # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
# tag_map = defaultdict(lambda : wn.NOUN)
# tag_map['J'] = wn.ADJ
# tag_map['V'] = wn.VERB
# tag_map['R'] = wn.ADV
# for index,entry in enumerate(data['comment']):
#     # Declaring Empty List to store the words that follow the rules for this step
#     Final_words = []
#     # Initializing WordNetLemmatizer()
#     word_Lemmatized = WordNetLemmatizer()
#     # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
#     for word, tag in pos_tag(entry):
#         # Below condition is to check for Stop words and consider only alphabets
#         if word not in stopwords.words('english') and word.isalpha():
#             word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
#             Final_words.append(word_Final)
#     # The final processed set of words for each iteration will be stored in 'text_final'
#     data.loc[index, 'comment_processed'] = str(Final_words)