from tqdm import tqdm

import os
from torchtext.vocab import GloVe,Vectors,Vocab
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split,Dataset

from torchtext.data.utils import get_tokenizer


import pickle
from collections import Counter
from urllib.request import urlopen
import io
import tarfile
import tempfile

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_list_to_file(lst, filename):
    """
    Save a list to a file using pickle serialization.

    Parameters:
        lst (list): The list to be saved.
        filename (str): The name of the file to save the list to.

    Returns:
        None
    """
    with open(filename, 'wb') as file:
        pickle.dump(lst, file)

def load_list_from_file(filename):
    """
    Load a list from a file using pickle deserialization.

    Parameters:
        filename (str): The name of the file to load the list from.

    Returns:
        list: The loaded list.
    """
    with open(filename, 'rb') as file:
        loaded_list = pickle.load(file)
    return loaded_list

tokenizer = get_tokenizer('basic_english')
def yield_tokens(dataiter):
    for _ , word in dataiter:
        yield tokenizer(word)



class GloVe_override(Vectors):
    url = {
        "6B": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/tQdezXocAJMBMPfUJx_iUg/glove-6B.zip",
    }

    def __init__(self, name="6B", dim=100, **kwargs) -> None:
        url = self.url[name]
        name = "glove.{}.{}d.txt".format(name, str(dim))
        #name = "glove.{}/glove.{}.{}d.txt".format(name, name, str(dim))
        super(GloVe_override, self).__init__(name, url=url, **kwargs)

class GloVe_override2(Vectors):
    url = {
        "6B": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/tQdezXocAJMBMPfUJx_iUg/glove-6B.zip",
    }

    def __init__(self, name="6B", dim=100, **kwargs) -> None:
        url = self.url[name]
        #name = "glove.{}.{}d.txt".format(name, str(dim))
        name = "glove.{}/glove.{}.{}d.txt".format(name, name, str(dim))
        super(GloVe_override2, self).__init__(name, url=url, **kwargs)

try:
    glove_embedding = GloVe_override(name="6B", dim=100)
except:
    try:
        glove_embedding = GloVe_override2(name="6B", dim=100)
    except:
        glove_embedding = GloVe(name="6B", dim=100)


counter = Counter(glove_embedding.stoi.keys()) # type: ignore
vocab = Vocab(counter, specials=['<unk>', '<pad>'])

def text_pipeline(text):
    tokens = tokenizer(text)
    ids = [vocab.stoi.get(tok, vocab.stoi['<unk>']) for tok in tokens] # type: ignore
    return ids

def label_pipeline(x):
   return int(x) 


urlopened = urlopen('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/35t-FeC-2uN1ozOwPs7wFg.gz')
tar = tarfile.open(fileobj=io.BytesIO(urlopened.read()))
tempdir = tempfile.TemporaryDirectory()
tar.extractall(tempdir.name)
tar.close()

class IMDBDataset(Dataset):
    def __init__(self, root_dir, train=True):
        """
        root_dir: The base directory of the IMDB dataset.
        train: A boolean flag indicating whether to use training or test data.
        """
        self.root_dir = os.path.join(root_dir, "train" if train else "test")
        self.neg_files = [os.path.join(self.root_dir, "neg", f) for f in os.listdir(os.path.join(self.root_dir, "neg")) if f.endswith('.txt')]
        self.pos_files = [os.path.join(self.root_dir, "pos", f) for f in os.listdir(os.path.join(self.root_dir, "pos")) if f.endswith('.txt')]
        self.files = self.neg_files + self.pos_files
        self.labels = [0] * len(self.neg_files) + [1] * len(self.pos_files)
        self.pos_inx=len(self.pos_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        return label, content
root_dir = tempdir.name + '/' + 'imdb_dataset'
train_iter = IMDBDataset(root_dir=root_dir, train=True)  # For training data
test_iter = IMDBDataset(root_dir=root_dir, train=False)  # For test dataart=train_iter.pos_inx

start=train_iter.pos_inx
start=0

imdb_label = {0: " negative review", 1: "positive review"}
num_class = len(set([label for (label, text) in train_iter ]))
num_train = int(len(train_iter) * 0.95)

# Randomly split the training dataset into training and validation datasets using `random_split`.
# The training dataset will contain 95% of the samples, and the validation dataset will contain the remaining 5%.
split_train_, split_valid_ = random_split(train_iter, [num_train, len(train_iter) - num_train])
def collate_batch(batch):
    label_list, text_list =[],[]
    for label, text in batch:
        label_list.append(label_pipeline(label))
        text_list.append(torch.tensor(text_pipeline(text), dtype=torch.long))
    
    label_list = torch.tensor(label_list,dtype=torch.long)
    text_list = pad_sequence(text_list, batch_first=True)
    return label_list.to(device), text_list.to(device)
BATCH_SIZE=32
train_loader = DataLoader(split_train_,batch_size=BATCH_SIZE,shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(split_train_,batch_size=BATCH_SIZE,shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_iter,batch_size=BATCH_SIZE,shuffle=True, collate_fn=collate_batch)
label,seqence=next(iter(valid_loader))

class TextClassifier(nn.Module):
    def __init__(self, num_classes, freeze=False) -> None:
        super(TextClassifier,self).__init__()
        self.embedding=nn.Embedding.from_pretrained(glove_embedding.vectors.to(device),freeze=freeze) # type: ignore
        self.fc1 = nn.Linear(in_features=100, out_features=128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
    def forward(self, x):
        x= self.embedding(x)
        x= torch.mean(x, dim=1)
        x= self.fc1(x)
        x= self.relu(x)
        x= self.fc2(x)
        return x
    
model=TextClassifier(num_classes=4,freeze=False)
model.to(device)
model.eval()
predicted_label=model(seqence)
def predict(text, model, text_pipeline):
    with torch.no_grad():
        text = torch.unsqueeze(torch.tensor(text_pipeline(text)),0).to(device)

        output = model(text)
        return imdb_label[output.argmax(1).item()]

def evaluate(dataloader,model,device):
    correct=0
    total=0
    for label, text in dataloader:
        label, text = label.to(device), text.to(device)
        predicted_label = model(text)
        _, predicted = torch.max(predicted_label.data, 1)
        total += label.size(0)
        correct = (predicted ==label).sum().item()
        acc = 100*correct/total
        return acc

print(evaluate(test_loader , model, device))



def train_model(model, optimizer, criterion, train_dataloader, valid_dataloader, epochs=100, model_name="my_modeldrop"):
    cum_loss_list = []
    acc_epoch = []
    best_acc = 0
    file_name = model_name

    for epoch in tqdm(range(1,epochs+1)):
        model.train()
        cum_loss = 0
        for _, (label,text) in enumerate(train_dataloader):
            optimizer.zero_grad()
            predicted_label = model(text)
            loss = criterion(predicted_label,label)
            cum_loss +=loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            loss.backward()
            optimizer.step()
        cum_loss_list.append(cum_loss)
        acc = evaluate(valid_dataloader,model=model,device=device)
        acc_epoch.append(acc)
        if acc > best_acc: # type: ignore
            best_acc = acc
            print(f"New best accuracy: {acc:.4f}")
    torch.save(model.state_dict(), "my_model.pth")
LR=1

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
model_name="model_imdb_freeze_true2"
train_model(model=model, optimizer=optimizer,criterion=criterion,train_dataloader=train_loader,valid_dataloader=valid_loader,epochs=4, model_name=model_name)
