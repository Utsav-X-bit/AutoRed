#!/usr/bin/env python3
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
EXP_DIR = ROOT_DIR / "experiment"

TRAIN_DATA = DATA_DIR / "strategy_predictor_train.jsonl"
FEATURE_VOCAB_PATH = EXP_DIR / "feature_vocab.json"
LABEL_VOCAB_PATH = EXP_DIR / "label_vocab.json"
MODEL_PATH = EXP_DIR / "strategy_predictor.pth"

# All 18 strategies
ATTACK_TYPES = [
    "instruction_leak",
    "trigger_phrase_discovery",
    "exception_discovery",
    "roleplay",
    "translation",
    "summarization",
    "system_prompt_recovery",
    "encoding_bypass",
    "markdown_smuggling",
    "latent_injection",
    "authority_override",
    "jailbreak_framing",
    "reflection_attack",
    "format_conversion",
    "json_smuggling",
    "yaml_smuggling",
    "base64_bypass",
    "unicode_bypass"
]

class StrategyDataset(Dataset):
    def __init__(self, data_path, feature_vocab, label_vocab):
        self.X = []
        self.y = []
        
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                
                # Build feature vector
                feat_vec = torch.zeros(len(feature_vocab))
                
                # Add primary type
                prim = f"primary:{item.get('primary_type', 'UNKNOWN')}"
                if prim in feature_vocab:
                    feat_vec[feature_vocab[prim]] = 1.0
                    
                # Add secondary flags
                for sec in item.get('secondary_flags', []):
                    sec_feat = f"secondary:{sec}"
                    if sec_feat in feature_vocab:
                        feat_vec[feature_vocab[sec_feat]] = 1.0
                        
                # Label
                strategy = item.get('strategy')
                if strategy in label_vocab:
                    self.X.append(feat_vec)
                    self.y.append(label_vocab[strategy])
                    
        self.X = torch.stack(self.X)
        self.y = torch.tensor(self.y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class StrategyPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def build_vocabs(data_path):
    features = set()
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            features.add(f"primary:{item.get('primary_type', 'UNKNOWN')}")
            for sec in item.get('secondary_flags', []):
                features.add(f"secondary:{sec}")
                
    feature_vocab = {f: i for i, f in enumerate(sorted(list(features)))}
    label_vocab = {l: i for i, l in enumerate(ATTACK_TYPES)}
    
    return feature_vocab, label_vocab

def main():
    print("Building vocabularies...")
    feature_vocab, label_vocab = build_vocabs(TRAIN_DATA)
    
    with open(FEATURE_VOCAB_PATH, "w") as f:
        json.dump(feature_vocab, f, indent=2)
    with open(LABEL_VOCAB_PATH, "w") as f:
        json.dump(label_vocab, f, indent=2)
        
    print(f"Feature vocab size: {len(feature_vocab)}")
    print(f"Label vocab size: {len(label_vocab)}")
    
    print("Loading dataset...")
    dataset = StrategyDataset(TRAIN_DATA, feature_vocab, label_vocab)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    
    model = StrategyPredictor(len(feature_vocab), len(label_vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            
        acc = correct / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.4f}")
        
    print(f"Saving model to {MODEL_PATH}")
    torch.save(model.state_dict(), MODEL_PATH)
    print("Done!")

if __name__ == "__main__":
    main()
