import os
import re
from collections import defaultdict, Counter
from tqdm import tqdm
import csv

data_path = "./dataset/tibetan_news_classification"

folders = os.listdir(data_path)
word_list = []

for folder in tqdm(folders):
    files = os.listdir(os.path.join(data_path, folder))
    for file in files:
        file_path = os.path.join(data_path, folder, file)
        with open(file_path, "r", encoding="utf-16") as f:
            for line in f:
                text = line.strip()
                words = re.split('་|༌|།|༎|༏|༐|༑', text)
                word_list.extend([word for word in words if word])
                
word_freq = Counter(word_list)

sorted_word_freq = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)

csv_file_path = "./word_frequency.csv"
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Word", "Frequency"])
    for word, freq in sorted_word_freq:
        writer.writerow([word, freq])
 
print(f"Sorted word frequencies have been saved to {csv_file_path}")

total_count = sum(word_freq.values())

filtered_words = []
covered_count = 0

for word, freq in sorted_word_freq:
    covered_count += freq
    filtered_words.append(word)
    if covered_count / total_count >= 0.999:
        break

special_tokens = ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"]

vocab = special_tokens + filtered_words
 
with open("syllable_vocab.txt", "w", encoding="utf-8") as f:
    for word in vocab:
        f.write(f"{word}\n")