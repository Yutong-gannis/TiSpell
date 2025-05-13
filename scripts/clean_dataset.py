import os
import re
from collections import Counter
from tqdm import tqdm
import csv
 

data_path = "./dataset/tibetan_news_classification"
output_path = "./dataset/cleaned_tibetan_news_classification"
vocab_file = "./tokenizer/syllable_vocab.txt"

with open(vocab_file, "r", encoding="utf-8") as f:
    vocab = set(word.strip() for word in f)

os.makedirs(output_path, exist_ok=True)

folders = os.listdir(data_path)

for folder in tqdm(folders, desc="Processing folders"):
    input_folder_path = os.path.join(data_path, folder)
    output_folder_path = os.path.join(output_path, folder)
    os.makedirs(output_folder_path, exist_ok=True)
    
    files = os.listdir(input_folder_path)
    for file in files:
        input_file_path = os.path.join(input_folder_path, file)
        output_file_path = os.path.join(output_folder_path, file)
        
        with open(input_file_path, "r", encoding="utf-16") as infile, \
             open(output_file_path, "w", encoding="utf-16") as outfile:
            
            for line in infile:
                text = line.strip()
                words = re.split('་|༌|།|༎|༏|༐|༑', text)
                if all(word in vocab for word in words if word):
                    outfile.write(line)

print("Data cleaning completed. Cleaned files are saved to:", output_path)