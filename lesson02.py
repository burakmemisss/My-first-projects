import os
import io
import tarfile
import math
import random
import time
from urllib.request import urlopen
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# torchtext tokenizer + vectors
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe, Vectors, Vocab


# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                       # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)            # even
        pe[:, 1::2] = torch.cos(position * div_term)            # odd
        pe = pe.unsqueeze(0)                                    # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, S, E)
        x = x + self.pe[:, : x.size(1), :] # pyright: ignore[reportIndexIssue]
        return self.dropout(x)


# -------------------------
# IMDB Dataset (folder-based)
# -------------------------
class IMDBDataset(Dataset):
    def __init__(self, root_dir: str, train: bool = True):
        """
        root_dir: base directory that contains imdb_dataset/
                 imdb_dataset/train/pos, imdb_dataset/train/neg ...
        train: True -> use train, False -> use test
        """
        self.root_dir = os.path.join(root_dir, "train" if train else "test")

        neg_dir = os.path.join(self.root_dir, "neg")
        pos_dir = os.path.join(self.root_dir, "pos")

        self.neg_files = [
            os.path.join(neg_dir, f)
            for f in os.listdir(neg_dir)
            if f.endswith(".txt")
        ]
        self.pos_files = [
            os.path.join(pos_dir, f)
            for f in os.listdir(pos_dir)
            if f.endswith(".txt")
        ]

        self.files = self.neg_files + self.pos_files
        self.labels = [0] * len(self.neg_files) + [1] * len(self.pos_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return label, text


# -------------------------
# Download & extract IMDB to a persistent folder
# -------------------------
def download_and_extract_imdb(data_root: str) -> str:
    """
    Downloads imdb_dataset tar.gz and extracts it into data_root.
    Returns the path to extracted imdb_dataset directory.
    """
    os.makedirs(data_root, exist_ok=True)

    imdb_root = os.path.join(data_root, "imdb_dataset")
    if os.path.isdir(imdb_root) and os.path.isdir(os.path.join(imdb_root, "train")):
        return imdb_root  # already extracted

    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/35t-FeC-2uN1ozOwPs7wFg.gz"
    print("Downloading IMDB dataset...")
    with urlopen(url) as resp:
        data = resp.read()

    print("Extracting IMDB dataset to:", data_root)
    tar = tarfile.open(fileobj=io.BytesIO(data))
    # Python 3.12+ tarfile güvenliği: filter='data'
    tar.extractall(data_root, filter="data")
    tar.close()

    if not os.path.isdir(imdb_root):
        raise RuntimeError(f"Extraction failed; imdb_dataset not found in: {data_root}")

    return imdb_root


# -------------------------
# GloVe loader (same idea as your override)
# -------------------------
class GloVe_override(Vectors):
    url = {
        "6B": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/tQdezXocAJMBMPfUJx_iUg/glove-6B.zip",
    }

    def __init__(self, name="6B", dim=100, **kwargs):
        url = self.url[name]
        fname = f"glove.{name}.{dim}d.txt"
        super().__init__(fname, url=url, **kwargs)


class GloVe_override2(Vectors):
    url = {
        "6B": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/tQdezXocAJMBMPfUJx_iUg/glove-6B.zip",
    }

    def __init__(self, name="6B", dim=100, **kwargs):
        url = self.url[name]
        fname = f"glove.{name}/glove.{name}.{dim}d.txt"
        super().__init__(fname, url=url, **kwargs)


def load_glove(dim=100):
    try:
        return GloVe_override(name="6B", dim=dim)
    except Exception:
        try:
            return GloVe_override2(name="6B", dim=dim)
        except Exception:
            return GloVe(name="6B", dim=dim)


# -------------------------
# Model
# -------------------------
class Net(nn.Module):
    def __init__(
        self,
        num_class: int,
        vocab,
        glove_embedding,
        freeze: bool = True,
        nhead: int = 2,
        dim_feedforward: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()

        embedding_dim = glove_embedding.dim  # 100

        # Build embedding matrix aligned with vocab indices
        embedding_matrix = torch.zeros((len(vocab), embedding_dim))
        for word, idx in vocab.stoi.items():
            if word in glove_embedding.stoi:
                g_idx = glove_embedding.stoi[word]
                embedding_matrix[idx] = glove_embedding.vectors[g_idx]

        self.emb = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze)

        self.pos_encoding = PositionalEncoding(d_model=embedding_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # ✅ (B, S, E) format
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(embedding_dim, num_class)

    def forward(self, x):
        # x: (B, S) token ids
        x = self.emb(x.long())              # (B, S, E)
        x = self.pos_encoding(x)            # (B, S, E)
        x = self.transformer_encoder(x)     # (B, S, E)
        x = x.mean(dim=1)                   # (B, E)
        x = self.classifier(x)              # (B, C)
        return x


# -------------------------
# Pipelines + collate
# -------------------------
def make_vocab_from_glove(glove_embedding):
    # GloVe vocabulary keys -> counter
    counter = Counter(glove_embedding.stoi.keys())
    vocab = Vocab(counter, specials=["<unk>", "<pad>"])
    return vocab


# -------------------------
# Eval + Train
# -------------------------
@torch.no_grad()
def evaluate(dataloader, model):
    model.eval()
    total_acc, total_count = 0, 0
    for label, text in dataloader:
        label, text = label.to(device), text.to(device)
        output = model(text)
        predicted = output.argmax(dim=1)
        total_acc += (predicted == label).sum().item()
        total_count += label.size(0)
    return total_acc / max(total_count, 1)


def train_model(
    model,
    optimizer,
    criterion,
    scheduler,
    train_dataloader,
    valid_dataloader,
    epochs=2,
    grad_clip=0.1,
):
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for label, text in tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs}"):
            label, text = label.to(device), text.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(text)
            loss = criterion(logits, label)

            loss.backward()  # ✅ FIX
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # ✅ step'ten önce

            optimizer.step()

            running_loss += loss.item()

        # ✅ LR scheduler: epoch sonunda
        if scheduler is not None:
            scheduler.step()

        val_acc = evaluate(valid_dataloader, model)
        avg_loss = running_loss / max(len(train_dataloader), 1)

        print(f"Epoch {epoch} | avg_loss={avg_loss:.4f} | val_acc={val_acc:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # 1) Persistent dataset folder (change if you want)
    DATA_ROOT = "/Users/burakmemis/Documents/datasets"  # ✅ kalıcı
    imdb_root = download_and_extract_imdb(DATA_ROOT)

    # 2) Build datasets
    train_dataset_full = IMDBDataset(root_dir=imdb_root, train=True)
    test_dataset = IMDBDataset(root_dir=imdb_root, train=False)

    # 3) Train/valid split + small train subset like your code
    num_train = int(len(train_dataset_full) * 0.95)
    split_train, split_valid = random_split(train_dataset_full, [num_train, len(train_dataset_full) - num_train])

    # keep only 5% of train split for speed
    num_small = max(1, int(len(split_train) * 0.05))
    split_train, _ = random_split(split_train, [num_small, len(split_train) - num_small])

    tokenizer = get_tokenizer("basic_english")

    glove_embedding = load_glove(dim=100)
    vocab = make_vocab_from_glove(glove_embedding)

    def text_pipeline(x: str):
        tokens = tokenizer(x)
        return [vocab.stoi.get(t, vocab.stoi["<unk>"]) for t in tokens] # type: ignore

    def collate_batch(batch):
        label_list, text_list = [], []
        for label, text in batch:
            label_list.append(label)
            text_list.append(torch.tensor(text_pipeline(text), dtype=torch.long))

        label_tensor = torch.tensor(label_list, dtype=torch.long)
        text_tensor = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"]) # type: ignore

        return label_tensor.to(device), text_tensor.to(device)

    BATCH_SIZE = 32
    train_loader = DataLoader(split_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(split_valid, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    # 4) Model
    num_class = 2
    model = Net(num_class=num_class, vocab=vocab, glove_embedding=glove_embedding, freeze=True, nhead=2, num_layers=2, max_len=512).to(device)

    # 5) Train config
    LR = 1.0  # senin değer; pratikte düşürmek daha stabil olur
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # 6) Train
    train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        epochs=2,
        grad_clip=0.1,
    )

    # 7) Test accuracy
    test_acc = evaluate(test_loader, model)
    print(f"Test accuracy: {test_acc:.4f}")

    # 8) Quick predict helper
    imdb_label = {0: "negative review", 1: "positive review"}

    @torch.no_grad()
    def predict(text: str):
        model.eval()
        ids = torch.tensor(text_pipeline(text), dtype=torch.long).unsqueeze(0).to(device)
        out = model(ids)
        return imdb_label[out.argmax(dim=1).item()]

    print("Predict:", predict("I like sports and stuff"))
