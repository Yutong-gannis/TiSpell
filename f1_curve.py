import os
import re
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from dataloader.corrupt import char_level, syllable_level, utils
from transformers import AutoTokenizer
from option import parse_args
from metrics import compute_precision_recall_f1
from model.tispell_roberta import TiSpell_RoBERTa
from model.tispell_cino import TiSpell_CINO
from model.roberta_bilstm import Roberta_BiLSTM


class TibetanDataset(Dataset):
    def __init__(self, args, tokenizer, possibilities):
        self.max_char_in_word = args.max_char_in_word
        self.max_char_length = args.max_char_length
        self.max_word_length = args.max_word_length
        self.max_subword_length = args.max_subword_length
        self.possibilities = possibilities
        self.tokenizer = tokenizer
        texts = []
        folders = os.listdir(args.data_path)
        for folder in tqdm(folders):
            files = os.listdir(os.path.join(args.data_path, folder))
            for file in files:
                with open(os.path.join(args.data_path, folder, file), "r", encoding="utf-16") as f:
                    for line in f:
                        text = line.strip()
                        if len(text) >= 20:
                            texts.append(line.strip())
        texts = texts[:args.num_sample]
        self.texts = texts
    
    def random_corrupt(self, text):
        syllable_list = utils.split_syllable(text)
        correct_syllable_list = syllable_list.copy()
        data = {'correct': '་'.join(correct_syllable_list)}
        for possibility in self.possibilities:
            syllable_list = utils.split_syllable(text)
            if random.random() < possibility:
                syllable_list, _ = syllable_level.syllable_random_delete(syllable_list)
            if random.random() < possibility:
                syllable_list, _ = syllable_level.syllable_random_exchange(syllable_list)
            if random.random() < possibility:
                syllable_list = syllable_level.syllable_random_merge(syllable_list)
            if random.random() < possibility:
                syllable_list = char_level.char_random_delete(syllable_list)
            if random.random() < possibility:
                syllable_list = char_level.char_random_insert(syllable_list)
            if random.random() < possibility:
                syllable_list = char_level.char_tall_short_replace(syllable_list)
            if random.random() < possibility:
                syllable_list = char_level.char_homomorphic_replace(syllable_list)
            if random.random() < possibility:
                syllable_list = char_level.char_inner_syllable_exchange(syllable_list)
            if random.random() < possibility:
                syllable_list = char_level.char_near_syllable_exchange(syllable_list)
            data[str(possibility)] = '་'.join(syllable_list)
        return data
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        text = self.random_corrupt(text)
            
        tokenized_text = {}
        for key, value in text.items():
            tokenized_text[key] = self.tokenizer(value, 
                                                 truncation=True, padding="max_length", max_length=self.max_subword_length, return_tensors="pt")
            
            tokenized_text[key]['input_ids'] = tokenized_text[key]['input_ids'].squeeze()
            tokenized_text[key]['attention_mask'] = tokenized_text[key]['attention_mask'].squeeze()
        return tokenized_text
    
def evaluation(args, model, tokenizer, test_loader, possibilities):
    fi_dict = {}
    f1_list_dict = {str(possibility): [] for possibility in possibilities}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            tokenized_text = batch
            correct_input_ids = tokenized_text['correct']['input_ids'].to(args.device)
            correct_texts = tokenizer.batch_decode(correct_input_ids, seq='་', skip_special_tokens=True)
            
            for key in f1_list_dict.keys():
                corrupt_input_ids = tokenized_text[key]['input_ids'].to(args.device)
                corrupt_attention_mask = tokenized_text[key]['attention_mask'].to(args.device)

                logit, _ = model(corrupt_input_ids, attention_mask=corrupt_attention_mask)
                pred_input_ids = torch.argmax(logit, dim=-1)
                pred_texts = tokenizer.batch_decode(pred_input_ids, seq='་', skip_special_tokens=True)

                for pred_text, correct_text in zip(pred_texts, correct_texts):
                    pred_text = pred_text.replace('་', ' ')
                    correct_text = correct_text.replace('་', ' ')
                    _, _, f1 = compute_precision_recall_f1(re.split(' ', pred_text), re.split(' ', correct_text))
                    f1_list_dict[key].append(f1)
    for key in f1_list_dict.keys():
        fi_dict[key] = sum(f1_list_dict[key])/len(f1_list_dict[key])
    return fi_dict


def main():
    args = parse_args()
    #possibilities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    possibilities = [0, 0.12, 0.20, 0.24, 0.28, 0.3]
    
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer/tibetan_roberta', use_fast=False, do_lower_case=False)
    
    dataset = TibetanDataset(args, tokenizer, possibilities)
    train_size = int(0.999 * len(dataset))
    val_size = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, val_size])

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    tispell_roberta = TiSpell_RoBERTa("./weights/tispell_roberta", tokenizer).to(args.device).eval()
    tispell_roberta.load_state_dict(torch.load("./weights/tispell_roberta/tispell_roberta.pth", map_location=args.device))
    tibetan_roberta_f1 = evaluation(args, tispell_roberta, tokenizer, test_loader, possibilities)
    print(tibetan_roberta_f1)
    plt.plot(tibetan_roberta_f1.keys(), tibetan_roberta_f1.values(), marker='o', label="tibetan_roberta")
    
    tispell_cino = TiSpell_CINO("./weights/tispell_cino", tokenizer).to(args.device).eval()
    tispell_cino.load_state_dict(torch.load("./weights/tispell_cino/tispell_cino.pth", map_location=args.device))
    tispell_cino_f1 = evaluation(args, tispell_roberta, tokenizer, test_loader, possibilities)
    plt.plot(tispell_cino_f1.keys(), tispell_cino_f1.values(), marker='o', label="tispell_cino")
    
    roberta_bilstm = Roberta_BiLSTM("./weights/roberta_bilstm", tokenizer).to(args.device).eval()
    roberta_bilstm.load_state_dict(torch.load("./weights/roberta_bilstm/roberta_bilstm.pth", map_location=args.device))
    roberta_bilstm_f1 = evaluation(args, tispell_roberta, tokenizer, test_loader, possibilities)
    plt.plot(roberta_bilstm_f1.keys(), roberta_bilstm_f1.values(), marker='o', label="roberta_bilstm")
    plt.xlabel("Possibility of Corruption Occurrence")
    plt.ylabel("F1-score")
    plt.legend()
    plt.grid()
    plt.savefig("f1_curve.png", dpi=300)

if __name__ == "__main__":
    main()
