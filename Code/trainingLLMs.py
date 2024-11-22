# Run the file: for BART -- nohup python -u finetune.py --model_type bart --summary_type general > OUTs/BART_FT_general.out &
# Run the file: for T5 -- nohup python -u finetune.py --model_type t5 --summary_type pd > OUTs/T5_FT_pd.out &

import os
import argparse
import numpy as np
import pandas as pd
import warnings
import random
import math
import re
from tqdm import tqdm
import gc

from collections import Counter

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from transformers.modeling_utils import PreTrainedModel, unwrap_model

from transformers import (
    AutoTokenizer,
    T5TokenizerFast,
    AutoModelForCausalLM,
    LlamaModel,
    T5EncoderModel,
    AdamW
)

from huggingface_hub import login
access_token = "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"
login(token = access_token)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:1")
    print("Using GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# DEVICE = torch.device("cpu")

# -------------------------------------------------------------- CONFIG -------------------------------------------------------------- #

def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEED = 42
set_random_seed(SEED)

TARGET_MAX_LEN = 1

MAX_EPOCHS = 10

EARLY_STOPPING_THRESHOLD = 2

SCALER = GradScaler()

input_label_sets = ['anecdotes', 'action_classification', 'consequence_classification', 'commonsense', 'deontology', 'justice', 'virtue', 'moralexceptqa', 'moralfoundationredditcorpus', 'storycommonsense', 'moralconvita', 'moralfoundationstwittercorpus']
input_labels_sets = ['moralintegritycorpus']
inputs_label_sets = ['dilemmas', 'utilitarianism', 'storal_en', 'storal_zh']

# ------------------------------------------------------------- DATA UTILS -------------------------------------------------------------- #

def read_csv_data(path):
    data = pd.read_csv(path)
    if 'labels' in data.columns:
        data.rename(columns = {'labels':'label'}, inplace = True)
    gc.collect()
    return data

def pad_seq(tensor: torch.tensor, dim: int, max_len: int):
    return torch.cat([tensor, torch.zeros(max_len - tensor.shape[0], dim)])

def get_multilabel_ready(label):
    lab = str(label)
    lab = re.sub("\[","", label)
    lab = re.sub("\]","", lab)
    lab = re.sub("\"","", lab)
    lab = re.sub("\'","", lab)
    lab = [x.strip() for x in lab.split(",")]
    return lab

def preprocess_dataset(text_path: str):
    dataset = read_csv_data(text_path)
    NUM_INP = 1

    if DATASET in input_label_sets:
        source = [SOURCE_PREFIX + str(s)for s in dataset['input'].tolist()]
        model_inputs = TOKENIZER(source, max_length=SOURCE_MAX_LEN, padding='max_length', truncation=True)
        
        all_labels = dataset['label'].tolist()
        idx2lab = list(set(all_labels))
        lab2idx = {k:v for v,k in enumerate(idx2lab)}

        wts = dict(Counter(all_labels))
        weights = torch.zeros((len(idx2lab)), dtype=torch.long).to(DEVICE)
        for k,v in wts.items():
            weights[lab2idx[k]] = max(wts.values())/v
        # print("weights: ", weights)

        target = [lab2idx[t] for t in dataset['label'].tolist()]

        model_inputs['input_ids'] = torch.tensor([i for i in model_inputs['input_ids']], dtype=torch.long, device=DEVICE)
        model_inputs['attention_mask'] = torch.tensor([a for a in model_inputs['attention_mask']], dtype=torch.long, device=DEVICE)
        model_inputs['labels'] = torch.tensor(target, dtype=torch.long, device=DEVICE)

    elif DATASET in input_labels_sets:
        source = [SOURCE_PREFIX + s for s in dataset['input'].tolist()]
        model_inputs = TOKENIZER(source, max_length=SOURCE_MAX_LEN, padding='max_length', truncation=True)

        all_labels = [get_multilabel_ready(x) for x in dataset['label']]
        idx2lab = list(set([x for xs in all_labels for x in xs]))
        lab2idx = {k:v for v,k in enumerate(idx2lab)}

        wts = dict(Counter(all_labels))
        weights = torch.zeros((len(idx2lab)), dtype=torch.long).to(DEVICE)
        for k,v in wts.items():
            weights[lab2idx[k]] = max(wts.values())/v
        # print("weights: ", weights)

        target = MultiLabelBinarizer().fit_transform(all_labels)

        model_inputs['input_ids'] = torch.tensor([i for i in model_inputs['input_ids']], dtype=torch.long, device=DEVICE)
        model_inputs['attention_mask'] = torch.tensor([a for a in model_inputs['attention_mask']], dtype=torch.long, device=DEVICE)
        model_inputs['labels'] = torch.tensor(target, dtype=torch.long, device=DEVICE)

    elif DATASET in inputs_label_sets:
        all_labels = dataset['label'].tolist()
        idx2lab = list(set(all_labels))
        lab2idx = {k:v for v,k in enumerate(idx2lab)}

        wts = dict(Counter(all_labels))
        weights = torch.zeros((len(idx2lab)), dtype=torch.float).to(DEVICE)
        for k,v in wts.items():
            weights[lab2idx[k]] = float(max(wts.values())/v)
        # print("weights: ", weights)

        target = [lab2idx[t] for t in dataset['label'].tolist()]

        sources_input_ids = []
        sources_attn_mask = []
        inp_cols = [x for x in dataset.columns if x != 'label' and "Unnamed" not in x]
        NUM_INP = len(inp_cols)
        for inp in inp_cols:
            source = [SOURCE_PREFIX + s for s in dataset[inp].tolist()]
            source = TOKENIZER(source, max_length=SOURCE_MAX_LEN, padding='max_length', truncation=True)
            source['input_ids'] = torch.tensor([i for i in source['input_ids']], dtype=torch.long, device=DEVICE)
            source['attention_mask'] = torch.tensor([a for a in source['attention_mask']], dtype=torch.long, device=DEVICE)
            sources_input_ids.append(source['input_ids'])
            sources_attn_mask.append(source['attention_mask'])
        
        model_inputs = {
            'input_ids': torch.stack(sources_input_ids, dim=1),     ## seq x num_inp x source_max_len
            'attention_mask': torch.stack(sources_attn_mask,dim=1),
            'labels': torch.tensor(target, dtype=torch.long, device=DEVICE)
        }

    del text_path 
    del dataset
    del source
    del all_labels
    del target
    gc.collect()

    return model_inputs, lab2idx, NUM_INP

def set_up_data_loader(text_path: str, set_type: str):
    dataset, lab2idx, NUM_INP = preprocess_dataset(text_path=text_path)
    dataset = TensorDataset(dataset['input_ids'], dataset['attention_mask'], dataset['labels'])

    if set_type == 'test':         ## No shuffling for test set
        return lab2idx, NUM_INP, DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return lab2idx, NUM_INP, DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def pad_to_max_len(tokenizer, tensor, max_length):
        if tokenizer is None:
            raise ValueError(
                f"Tensor need to be padded to `max_length={max_length}` but no tokenizer was passed when creating "
                "this `Trainer`. Make sure to create your `Trainer` with the appropriate tokenizer."
            )
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        padded_tensor = pad_token_id * torch.ones((tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device)
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

def get_scores(reference_list: list, hypothesis_list: list):
    acc = accuracy_score(reference_list, hypothesis_list)
    report = classification_report(reference_list, hypothesis_list)

    return {"accuracy": acc, "report": report}

def _save(model, output_dir: str, tokenizer=None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(model, PreTrainedModel):
            if isinstance(unwrap_model(model), PreTrainedModel):
                if state_dict is None:
                    state_dict = model.state_dict()
                unwrap_model(model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, DATASET))
        else:
            model.save_pretrained(output_dir, state_dict=state_dict)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

def save_model(model, output_dir: str, tokenizer=None, state_dict=None):
        _save(model,output_dir, tokenizer=tokenizer, state_dict=state_dict)

def load_model(model, input_dir: str):
    state_dict = torch.load(input_dir, map_location = DEVICE)
    model.load_state_dict(state_dict)
    return model

# ----------------------------------------------------- MODEL ----------------------------------------------------- #

class MyLlamaModel(nn.Module):
    def __init__(self, pretrained_model, num_inp, num_classes = 2):
        super().__init__()
        self.encoder = LlamaModel.from_pretrained(pretrained_model)
        for (name, param) in self.encoder.named_parameters():
            if "layers.31" not in name and "layers.30" not in name and "layers.29" not in name and "layers.28" not in name and name != "norm.weight":
                param.requires_grad = False
        for (name, param) in self.encoder.named_parameters():
            print(name, param.requires_grad)

        self.ln1 = nn.LayerNorm(4096)
        self.linear1 = nn.Linear(4096, 768)
        self.ln2 = nn.LayerNorm(768)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.linear2 = nn.Linear(num_inp*768, num_classes)
        
    def forward(self, source):
        if DATASET in input_label_sets:
            op = self.encoder(source['input_ids'], source['attention_mask']).last_hidden_state.mean(dim=1)
            op = self.linear2(self.dropout(self.linear1(op)))

        elif DATASET in inputs_label_sets:
            op = []
            source['input_ids'] = torch.transpose(source['input_ids'], 0, 1)
            source['attention_mask'] = torch.transpose(source['attention_mask'], 0, 1)
            for input_ids, attention_mask in zip(source['input_ids'], source['attention_mask']):
                op.append(self.ln2(self.linear1(self.ln1(self.encoder(input_ids, attention_mask).last_hidden_state.mean(dim=1)))))
            op = torch.cat(op, dim = 1)
            op = self.linear2(self.dropout(op))

        elif DATASET in input_labels_sets:
            op = self.encoder(source['input_ids'], source['attention_mask']).last_hidden_state.mean(dim=1)
            op = self.linear2(self.dropout(self.linear1(op)))

        for item in source.items():
            del item
        del source
        gc.collect()
        return op
    
# ----------------------------------------------------- TRAINING UTILS ----------------------------------------------------- #

def prepare_for_training(model, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    return optimizer, criterion

def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    epoch_train_loss = 0.0
    for step, batch in enumerate(tqdm(data_loader, desc="Training Iteration")):
        input_ids, attention_mask, target = batch

        source = {
            'input_ids': input_ids.to(DEVICE),
            'attention_mask': attention_mask.to(DEVICE)
        }

        target = target.to(DEVICE)
        optimizer.zero_grad()
        logits = model(source)
        
        loss = criterion(logits, target)
        epoch_train_loss += loss.item()

        loss.backward()
        optimizer.step()

        del loss
        del input_ids
        del attention_mask
        del logits
        del target
        for item in source.items():
            del item
        del source
        gc.collect()
        torch.cuda.empty_cache()

        # if step == 2:
        #     break

    del batch
    gc.collect()
    torch.cuda.empty_cache()
    
    return epoch_train_loss/ step

def val_epoch(model, data_loader, criterion):
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc="Validation Iteration")):
            input_ids, attention_mask, target = batch
            source = {
                'input_ids': input_ids.to(DEVICE),
                'attention_mask': attention_mask.to(DEVICE)
            }
            target = target.to(DEVICE)

            logits = model(source)

            loss = criterion(logits, target)
            epoch_val_loss += loss.item()  

            del loss
            del input_ids
            del attention_mask
            del logits
            del target
            for item in source.items():
                del item
            del source
            gc.collect()
            torch.cuda.empty_cache()
            # if step == 2:
            #     break

    del batch
    gc.collect()
    torch.cuda.empty_cache() 
    
    return epoch_val_loss/ step

def test_epoch(model, data_loader, desc):
    model.eval()
    predictions = []
    gold = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc=desc)):
            input_ids, attention_mask, target = batch
            source = {
                'input_ids': input_ids.to(DEVICE),
                'attention_mask': attention_mask.to(DEVICE)
            }

            target = target.to(DEVICE)
            logits = model(source)

            if DATASET in input_labels_sets:
                predicted_classes = nn.Softmax(dim=1)(logits)
                predictions.append(predicted_classes.tolist())
            else:
                predicted_classes = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
                predictions.extend(predicted_classes.tolist())

            gold.extend(target.tolist())
            del input_ids
            del attention_mask
            del logits
            del target
            for item in source.items():
                del item
            del source
            gc.collect()
            torch.cuda.empty_cache()
            # if step == 2:
            #     break

    del batch
    gc.collect()
    torch.cuda.empty_cache()
    return predictions, gold

def compute_metrics(model, tokenizer,data_loader, desc):

    predictions, gold = test_epoch(model, data_loader, desc=desc)
    result = get_scores(gold, predictions)

    torch.cuda.empty_cache() 
    
    return predictions, gold, result  

def train(model, tokenizer, train_data_loader, val_data_loader, test_data_loader, learning_rate, model_type):
    train_losses = []
    val_losses = []
    
    optimizer, criterion = prepare_for_training(model=model, learning_rate=learning_rate)
    
    patience = 0
    min_val_loss = 99999
    bad_loss = 0
    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_data_loader, optimizer, criterion)
        train_losses.append(train_loss)
        print("Epoch: {}\ttrain_loss: {}".format(epoch+1, train_loss))

        val_loss = val_epoch(model, val_data_loader, criterion)
        val_losses.append(val_loss)

        if val_loss < min_val_loss:
            bad_loss = 0
            min_val_loss = val_loss
            test_pred, test_gold, test_results = compute_metrics(model, tokenizer, test_data_loader, desc="Test Iteration")
            
            print("Epoch: {}\ttrain_loss: {}\tval_loss: {}\tmin_val_loss: {}".format(epoch+1, train_loss, val_loss, min(val_losses)))
            print("\nTest Accuracy: ", test_results['accuracy'])
            print("\nTest Classification Report:\n", test_results['report'])
        
            path = OUTPUT_DIR + DATASET + "_" + LOAD_MODEL
            save_model(model, path, tokenizer)
            print("Model saved at path: ", path)

            test_df = pd.DataFrame()
            test_df['Predictions'] = test_pred
            test_df['Gold'] = test_gold

            csv_path = OUTPUT_DIR + model_type + "_"  + DATASET + "_" + LOAD_MODEL + '.csv'
            test_df.to_csv(csv_path)
            print("Predictions saved at path: ", csv_path)
        else:
            bad_loss += 1
        
        if bad_loss == EARLY_STOPPING_THRESHOLD:
            print("Stopping early...")
            break

        torch.cuda.empty_cache()
    torch.cuda.empty_cache()

def infer(model, tokenizer, test_data_loader, model_type, save_csv=False):
    test_pred, test_gold, test_results = compute_metrics(model, tokenizer, test_data_loader, desc="Test Iteration")
            
    print("\nTest Accuracy: ", test_results['accuracy'])
    print("\nTest Classification Report:\n", test_results['report'])

    if save_csv:
        test_df = pd.DataFrame()
        test_df['Predictions'] = test_pred
        test_df['Gold'] = test_gold

        csv_path = OUTPUT_DIR + model_type + "_"  + DATASET + "_" + LOAD_MODEL + '.csv'
        test_df.to_csv(csv_path)
        print("Predictions saved at path: ", csv_path)

    torch.cuda.empty_cache()

# ------------------------------------------------------------ MAIN MODEL ------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='llama_3', type=str, help='Choose from [llama-3]')
    parser.add_argument('--file_path', default='./', type=str, help='The path containing train/val/test.csv')
    parser.add_argument('--batch_size', default=12, type=int, help='The batch size for training')
    parser.add_argument('--load_model', default='None', type=str, help='The path containing the pretrained model to load. None if no model.')
    parser.add_argument('--only_eval', default='FALSE', type=str, help='Set to "TRUE" if only have to perform evaluation on test set.')
    args = parser.parse_args()

    TEXT_INPUT_PATH = "/".join(args.file_path.split("/")[:-1])
    DATASET = args.file_path.split("/")[-2]
    LOAD_MODEL = ""
    BATCH_SIZE = args.batch_size
    
    if args.model_type == 'llama_3':
        SOURCE_MAX_LEN = 1024
        SOURCE_PREFIX = ''
        print("Using Llama 3")
        if args.load_model != "None":
            TOKENIZER = AutoTokenizer.from_pretrained("/".join(args.load_model.split("/")[:-1]), truncation_side='left')
            LOAD_MODEL = args.load_model.split("/")[-1]
        else:
            TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", truncation_side='left')
        TOKENIZER.pad_token = TOKENIZER.eos_token
        print("Llama 3 Tokenizer loaded...\n")
    elif args.model_type == 'flan-t5':
        print("Using Flan-T5")
        TOKENIZER = T5TokenizerFast.from_pretrained('google/flan-t5-base', truncation_side='left')
        print("Flan-T5 Tokenizer loaded...\n")
        SOURCE_MAX_LEN = 1024
        SOURCE_PREFIX = 'Classify: '
    else:
        print("Error: Wrong model type")
        exit(0)
    # ------------------------------ READ DATASET ------------------------------ #
    
    lab2idx, NUM_INP, train_dataset = set_up_data_loader(text_path = TEXT_INPUT_PATH + '/train.csv', set_type = 'train')
    if "val.csv" in os.listdir(args.file_path):
        _, __, val_dataset = set_up_data_loader(text_path = TEXT_INPUT_PATH + '/val.csv', set_type = 'val')
    else:
        _, __, val_dataset = set_up_data_loader(text_path = TEXT_INPUT_PATH + '/test.csv', set_type = 'val')
    _, __, test_dataset = set_up_data_loader(text_path = TEXT_INPUT_PATH + '/test.csv', set_type = 'test')
    
    # ------------------------------ MODEL SETUP ------------------------------ #

    if args.model_type == 'llama_3':
        MODEL = MyLlamaModel("meta-llama/Meta-Llama-3-8B", num_inp=NUM_INP, num_classes=len(lab2idx.keys()))
        print("Llama Model loaded...\n")
        print(MODEL)
        OUTPUT_DIR = "./models/llama_3/llama_3/"
        if args.load_model != "None":
            print("Loading model ", args.load_model, "...")
            MODEL = load_model(MODEL, args.load_model)
            print("Model State Dict Loaded")
        LEARNING_RATE = 5e-6
    
    else:
        print("Error: Wrong model type")
        exit(0)

    MODEL.to(DEVICE)
    print(LEARNING_RATE)
    print(OUTPUT_DIR)
    print(SOURCE_PREFIX)
    
    # ------------------------------ TRAINING SETUP ------------------------------ #
    if args.only_eval == "TRUE":
        infer(model=MODEL,
            tokenizer=TOKENIZER,
            test_data_loader=test_dataset,
            model_type=args.model_type,
            save_csv=True
        )
    else:
        train(model=MODEL,
            tokenizer=TOKENIZER,
            train_data_loader=train_dataset,
            val_data_loader=val_dataset,
            test_data_loader=test_dataset,
            learning_rate=LEARNING_RATE,
            model_type=args.model_type
        )
    
    print("Model Trained!")