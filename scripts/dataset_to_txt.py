import os
import re
import random
from tqdm import tqdm

data_path = "./dataset/cleaned_tibetan_news_classification"
output_file = "./tibetan_news_classification.txt"
alphabet_file = "/home/lyt/JamSpell/data/alphabet_ti.txt"

with open(alphabet_file, "r", encoding="utf-8") as f:
    for line in f:
        alphabet = line.strip()
        break
                    
folders = os.listdir(data_path)

num_samples = 0
all_texts = []  # 用于保存所有符合条件的句子

# 遍历文件夹并读取文件
for folder in tqdm(folders):
    folder_path = os.path.join(data_path, folder)
    files = os.listdir(folder_path)
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        
        with open(file_path, "r", encoding="utf-16") as f:
            for line in f:
                text = line.strip()
                #if len(text) >= 20:  # 只保留长度大于等于20的句子
                    #text = text.replace("་", " ")
                    #text = text.replace("༌", " ")
                    #text = text.replace("།", ".")
                    #text = text.replace("༎", ".")
                    #text = text.replace("༏", ".")
                    #text = text.replace("༐", ".")
                    #text = text.replace("༑", ".")
                    #words = re.split(' ', text)
                    #words = re.split('་|༌|།|༎|༏|༐|༑', text)
                    #for i in reversed(range(len(words))):
                    #    for x in words[i]:
                    #        if x not in alphabet and x != ' ':
                    #            del words[i]
                all_texts.append(text)  # 将符合条件的句子加入列表
                num_samples += 1

# 如果总句子数超过1000000，随机选择1000000个句子
#if num_samples > 1000000:
#    selected_texts = random.sample(all_texts, 1000000)
#else:
selected_texts = all_texts

# 写入到文件
with open(output_file, "w", encoding="utf-8") as out_file:
    for text in selected_texts:
        out_file.write(text + "\n")
            

print(f"所有行已成功合并到 {output_file}")
