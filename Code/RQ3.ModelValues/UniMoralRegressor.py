import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, ModernBertForSequenceClassification
from datasets import Dataset
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
import os

act2lab = {'1': 0, '2': 1, 1: 0, 2: 1, 'A': 0, 'B': 1, 'a': 0, 'b': 1}

df1 = pd.read_csv("/home/kshivan/MoralxNLP/UniMoral/Final_data/English_long_formatted.csv")
df2 = pd.read_csv("/home/kshivan/MoralxNLP/UniMoral/Final_data/English_short_formatted.csv")
common_cols = df1.columns.intersection(df2.columns)
df = pd.concat([df1[common_cols], df2[common_cols]], ignore_index=True)
df = df.dropna()
df["input_text"] = df.apply(lambda row: f"{row['Scenario'].strip()} Therefore, the person decides to {ast.literal_eval(row['Possible_actions'])[act2lab[row['Selected_action']]].strip()}.", axis=1)

cultural_dims = ["Care", "Equality", "Proportionality", "Loyalty", "Authority", "Purity"]
labels = []
for c_val in df["Moral_values"]:
    c_val = ast.literal_eval(c_val)
    lab = [
        c_val['Care'],
        c_val['Equality'],
        c_val['Proportionality'],
        c_val['Loyalty'],
        c_val['Authority'],
        c_val['Purity']
    ]
    labels.append(lab)

labels = np.array(labels, dtype=float)
min_vals = labels.min(axis=0)
max_vals = labels.max(axis=0)
labels = (labels - min_vals) / (max_vals - min_vals)
labels = labels * 2 - 1     ## Normalize between -1 and 1 instead of 0 and 1

for i, dim in enumerate(cultural_dims):
    df[dim] = labels[:, i]

dataset = Dataset.from_pandas(df[["input_text"] + cultural_dims])
print(dataset)

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

def tokenize(example):
    return tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=256)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"] + cultural_dims)

def format_dataset(example):
    return {
        "input_ids": example["input_ids"],
        "attention_mask": example["attention_mask"],
        "labels": torch.tensor([example[dim] for dim in cultural_dims], dtype=torch.float)
    }

dataset = dataset.map(format_dataset)
dataset = dataset.train_test_split(test_size=0.1)
print(dataset)

class BertForCulturalRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = ModernBertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=6)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits, labels)
        
        return {"loss": loss, "logits": logits}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 8
learning_rate = 5e-8
num_epochs = 500

output_dir = "./MoralClassifier"
os.makedirs(output_dir, exist_ok=True)

train_dataloader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False)

model = BertForCulturalRegression()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / num_batches

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            total_loss += loss.item()
            num_batches += 1
            
            all_predictions.append(logits.cpu())
            all_labels.append(labels.cpu())
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    mae = torch.mean(torch.abs(all_predictions - all_labels)).item()
    
    r2_scores = []
    for i in range(all_predictions.shape[1]):
        pred_dim = all_predictions[:, i]
        label_dim = all_labels[:, i]
        
        ss_res = torch.sum((label_dim - pred_dim) ** 2)
        ss_tot = torch.sum((label_dim - torch.mean(label_dim)) ** 2)
        r2 = 1 - ss_res / ss_tot
        r2_scores.append(r2.item())
    
    avg_r2 = np.mean(r2_scores)
    
    return total_loss / num_batches, mae, avg_r2, r2_scores

best_val_loss = float("inf")
best_model_path = None

print("Starting training...")
train_losses, eval_losses, eval_r2, eval_mae = [], [], [], []
patience = 0

best_model = model
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print("-" * 30)
    
    train_loss = train_epoch(model, train_dataloader, optimizer, device)
    print(f"Training Loss: {train_loss:.4f}")
    
    eval_loss, mae, avg_r2, r2_scores = evaluate(model, eval_dataloader, device)
    print(f"Evaluation Loss: {eval_loss:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Average R² Score: {avg_r2:.4f}")
    print("R2 Scores by dimension:")
    for i, (dim, score) in enumerate(zip(cultural_dims, r2_scores)):
        print(f"  {dim}: {score:.4f}")

    train_losses.append(train_loss)
    eval_losses.append(eval_loss)
    eval_r2.append(avg_r2)
    eval_mae.append(mae)

    print(train_loss, eval_loss, avg_r2, mae)
    
    if eval_loss < best_val_loss:
        patience = 0
        best_val_loss = eval_loss
        best_model = model
        print(f"Best model updated and saved to {best_model_path}")
    else:
        patience += 1
        print("No improvement this epoch.")

    if patience >= 5:
        break

best_model_path = os.path.join(output_dir, "best_model.pt")
torch.save({
    'epoch': epoch + 1,
    'model_state_dict': best_model.state_dict(),
    # 'optimizer_state_dict': optimizer.state_dict(),
}, best_model_path)

print("\nTraining completed!")
print(f"Best model was saved at epoch {epoch} at: {best_model_path}")

df = pd.DataFrame()
df['train_losses'] = train_losses
df['eval_losses'] = eval_losses
df['eval_r2'] = eval_r2
df['eval_mae'] = eval_mae

df.to_csv(f"{output_dir}/train_stats.csv")

# import torch
# import torch.nn as nn
# from transformers import ModernBertForSequenceClassification
# import os

# # Define model wrapper
# class BertForCulturalRegression(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bert = ModernBertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=6)
    
#     def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         logits = outputs.logits

#         loss = None
#         if labels is not None:
#             loss_fn = torch.nn.MSELoss()
#             loss = loss_fn(logits, labels)
        
#         return {"loss": loss, "logits": logits}

# # --- Load model checkpoint ---
# output_dir = "./CultureClassifier"
# checkpoint_path = os.path.join(output_dir, "best_model.pt")

# # Initialize and load model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = BertForCulturalRegression().to(device)

# checkpoint = torch.load(checkpoint_path, map_location=device)
# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval()

# print(f"Loaded model from epoch {checkpoint['epoch']} with eval loss {checkpoint['eval_loss']:.4f}")