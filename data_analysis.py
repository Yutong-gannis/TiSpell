import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd


print("Tibetan news classification")
texts = []
data_path = "./dataset/cleaned_tibetan_news_classification"
folders = os.listdir(data_path)
for folder in folders:
    num_sentences = 0
    total_syllable = 0
    total_character = 0
    syllable_lengths = []
    print(folder)
    files = os.listdir(os.path.join(data_path, folder))
    for file in files:
        with open(os.path.join(data_path, folder, file), "r", encoding="utf-16") as f:
            for line in f:
                text = line.strip()
                if len(text) >= 20:
                    num_sentences = num_sentences + 1
                    total_syllable = total_syllable + len(re.split('་|༌|།|༎|༏|༐|༑', text))
                    total_character = total_character + len(text)
                    syllable_lengths.append(len(re.split('་|༌|།|༎|༏|༐|༑', text)))
                    texts.append(line.strip())
    mean_character = total_character / num_sentences
    mean_syllable = total_syllable / num_sentences
    
    percentile_75 = np.percentile(syllable_lengths, 75)
    percentile_95 = np.percentile(syllable_lengths, 95)
    
    print("num_sentences: ", num_sentences)
    print("mean_character: ", mean_character)
    print("mean_syllable: ", mean_syllable)
    print("75th percentile of syllable length: ", percentile_75)
    print("95th percentile of syllable length: ", percentile_95)

print("TUSA")
texts_pos = []
texts_neg = []
data_path = "./dataset/TUSA"
files = os.listdir(data_path)

for file in files:
    texts = pd.read_csv(os.path.join(data_path, file))
    for i in range(len(texts)):
        text = texts.iloc[i]["text"].strip()
        label = texts.iloc[i]["label"]
        if len(text) >= 20:
            if label == 1:
                texts_pos.append(text)
            else:
                texts_neg.append(text)
      
for i in range(len([texts_neg, texts_pos])):
    print(i)
    total_syllable = 0
    total_character = 0
    syllable_lengths = []
    texts = [texts_pos, texts_neg][i]
    num_sentences = len(texts)
    for text in texts:
        num_syllable = len(re.split('་|༌|།|༎|༏|༐|༑', text))
        total_syllable = total_syllable + num_syllable
        total_character = total_character + len(text)
        syllable_lengths.append(num_syllable)
            
    mean_character = total_character / num_sentences
    mean_syllable = total_syllable / num_sentences
    
    percentile_75 = np.percentile(syllable_lengths, 75)
    percentile_95 = np.percentile(syllable_lengths, 95)
    
    print("num_sentences: ", num_sentences)
    print("mean_character: ", mean_character)
    print("mean_syllable: ", mean_syllable)
    print("75th percentile of syllable length: ", percentile_75)
    print("95th percentile of syllable length: ", percentile_95)
    