import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
import numpy as np
import generate_dicts as gd
import textClean as tc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# get embeddings array and vocab dictionary
embs_npa = gd.getEmbs()
vocab_npa_dict = gd.getVoc()

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
    data.at[i, 'text'] = tc.clean_text(data.iloc[i]['text'])
    data.at[i, 'text'] = data.at[i, 'text'].lower()
    data.at[i, 'text'] = word_tokenize(data.at[i, 'text'])
    data.at[i, 'sentiment'] = sentiment_map[data.at[i, 'sentiment']]
data.to_json('train-balanced-sarcasm-processed.json')

# data = pd.read_json("train-balanced-sarcasm-processed.json")
data_len = len(data)

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
    def __init__(self, embedding_dim, embs_npa) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float()).to(device)
        assert self.embedding.weight.shape == embs_npa.shape
        self.lstm = nn.LSTM(embedding_dim, 196, bidirectional=True, batch_first=True, device=device, dropout=0.5, num_layers=2)
        self.linear_1 = nn.Linear(196*2, 3, device=device)
    def forward(self, x):
        embed_full = self.embedding(x)
        lstm_output, _ = self.lstm(embed_full)
        max_pooled = torch.max(lstm_output, axis=1).values.view(BATCH_SIZE, 196*2)
        output = nn.functional.sigmoid(self.linear_1(max_pooled))
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

model = SarcasmModel(embedding_dims, embs_npa)
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

    torch.save(model.state_dict(), "trained_model.pt")

loss_graph.plot(x="id", y="total loss")
acc_graph.plot(x="id", y="train acc")
acc_graph.plot(x="id", y="val acc")