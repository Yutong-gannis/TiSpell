import os
import re
from tqdm import tqdm
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .corrupt import char_level, syllable_level, utils
from .corrupt_ratio import *
import json


class TibetanDatasetOneTokenizer(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        self.mode = mode
        data_path = args.data_path
        num_sample = args.num_sample
        self.max_char_in_word = args.max_char_in_word
        self.max_char_length = args.max_char_length
        self.max_word_length = args.max_word_length
        self.max_subword_length = args.max_subword_length
        texts = []
        folders = os.listdir(data_path)
        for folder in tqdm(folders):
            files = os.listdir(os.path.join(data_path, folder))
            for file in files:
                with open(os.path.join(data_path, folder, file), "r", encoding="utf-16") as f:
                    for line in f:
                        text = line.strip()
                        if len(text) >= 20:
                            texts.append(line.strip())
        print("origin sample: ", len(texts))
        #print("max character length: ", max(len(text) for text in texts))
        #print("max syllable length: ", max(len(re.split('་|༌|།|༎|༏|༐|༑', text)) for text in texts))
        #print("95 persent character length: ", np.percentile([len(text) for text in texts], 95))
        #print("95 persent syllable length: ", np.percentile([len(re.split('་|༌|།|༎|༏|༐|༑', text)) for text in texts], 95))
        self.texts = texts[:num_sample]
        self.tokenizer = tokenizer
    
    def random_corrupt(self, text):
        syllable_list = utils.split_syllable(text)
        correct_syllable_list = syllable_list.copy()
        if random.random() < syllable_random_delete_possibility:
            syllable_list, syllable_list_mask = syllable_level.syllable_random_delete(syllable_list)
        else:
            syllable_list_mask = syllable_list
        if random.random() < syllable_random_exchange_possibility:
            syllable_list, _ = syllable_level.syllable_random_exchange(syllable_list)
        if random.random() < syllable_random_merge_possibility:
            syllable_list = syllable_level.syllable_random_merge(syllable_list)
        if random.random() < char_random_delete_possibility:
            syllable_list = char_level.char_random_delete(syllable_list)
        if random.random() < char_random_insert_possibility:
            syllable_list = char_level.char_random_insert(syllable_list)
        if random.random() < char_tall_short_replace_possibility:
            syllable_list = char_level.char_tall_short_replace(syllable_list)
        if random.random() < char_homomorphic_replace_possibility:
            syllable_list = char_level.char_homomorphic_replace(syllable_list)
        if random.random() < char_inner_syllable_exchange_possibility:
            syllable_list = char_level.char_inner_syllable_exchange(syllable_list)
        if random.random() < char_near_syllable_exchange_possibility:
            syllable_list = char_level.char_near_syllable_exchange(syllable_list)
        text = {'correct': '་'.join(correct_syllable_list),
                'random_corrupt':  '་'.join(syllable_list),
                'mask':  '་'.join(syllable_list_mask),
                }
        return text
    
    def single_corrupt(self, text):
        syllable_list = utils.split_syllable(text)
        correct_list = syllable_list.copy()
        char_random_delete_list = char_level.char_random_delete(syllable_list)
        char_random_insert_list = char_level.char_random_insert(syllable_list)
        char_tall_short_replace_list = char_level.char_tall_short_replace(syllable_list)
        char_homomorphic_replace_list = char_level.char_homomorphic_replace(syllable_list)
        char_inner_syllable_replace_list = char_level.char_inner_syllable_exchange(syllable_list)
        char_near_syllable_replace_list = char_level.char_near_syllable_exchange(syllable_list)
        
        syllable_random_delete_list, _ = syllable_level.syllable_random_delete(syllable_list)
        syllable_random_exchange_list, _ = syllable_level.syllable_random_exchange(syllable_list)
        syllable_random_merge_list = syllable_level.syllable_random_merge(syllable_list)
        
        if random.random() < char_random_delete_possibility:
            syllable_list = char_level.char_random_delete(syllable_list)
        if random.random() < char_random_insert_possibility:
            syllable_list = char_level.char_random_insert(syllable_list)
        if random.random() < char_tall_short_replace_possibility:
            syllable_list = char_level.char_tall_short_replace(syllable_list)
        if random.random() < char_homomorphic_replace_possibility:
            syllable_list = char_level.char_homomorphic_replace(syllable_list)
        if random.random() < char_inner_syllable_exchange_possibility:
            syllable_list = char_level.char_inner_syllable_exchange(syllable_list)
        if random.random() < char_near_syllable_exchange_possibility:
            syllable_list = char_level.char_near_syllable_exchange(syllable_list)
        if random.random() < syllable_random_delete_possibility:
            syllable_list, _ = syllable_level.syllable_random_delete(syllable_list)
        if random.random() < syllable_random_exchange_possibility:
            syllable_list, _ = syllable_level.syllable_random_exchange(syllable_list)
        if random.random() < syllable_random_merge_possibility:
            syllable_list = syllable_level.syllable_random_merge(syllable_list)
        text = {'correct': '་'.join(correct_list),
                'random_corrupt': '་'.join(syllable_list),
                'char_random_delete':  '་'.join(char_random_delete_list),
                'char_random_insert':  '་'.join(char_random_insert_list),
                'char_tall_short_replace':  '་'.join(char_tall_short_replace_list),
                'char_homomorphic_replace':  '་'.join(char_homomorphic_replace_list),
                'char_inner_syllable_exchange':  '་'.join(char_inner_syllable_replace_list),
                'char_near_syllable_exchange':  '་'.join(char_near_syllable_replace_list),
                'syllable_random_delete':  '་'.join(syllable_random_delete_list),
                'syllable_random_exchange':  '་'.join(syllable_random_exchange_list),
                'syllable_random_merge':  '་'.join(syllable_random_merge_list),}
        return text
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.mode == 'train':
            text = self.random_corrupt(text)
        else:
            text = self.single_corrupt(text)
            
        tokenized_text = {}
        for key, value in text.items():
            tokenized_text[key] = self.tokenizer(value, 
                                                 truncation=True, padding="max_length", max_length=self.max_subword_length, return_tensors="pt")
            
            tokenized_text[key]['input_ids'] = tokenized_text[key]['input_ids'].squeeze()
            tokenized_text[key]['attention_mask'] = tokenized_text[key]['attention_mask'].squeeze()
        return tokenized_text
