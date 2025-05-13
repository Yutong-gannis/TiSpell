import torch
import numpy as np
from transformers import BertTokenizer


class Tokenizer:
    def __init__(self, vocab_file: str = None, vocab: list = None, padding_token: str = "<PAD>"):
        if vocab_file:
            self.vocab = self.load_vocab(vocab_file)
        elif vocab:
            self.vocab = vocab
        else:
            raise ValueError("please provide file or list")
        
        if padding_token not in self.vocab:
            self.vocab.append(padding_token)
        self.padding_token = padding_token
        self.vocab_size = len(self.vocab)
        self.token_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_token = {idx: char for idx, char in enumerate(self.vocab)}
    
    def load_vocab(self, vocab_file: str) -> list:
        vocab = []
        try:
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"File {vocab_file} not found")
        except Exception as e:
            print(f"Error: {e}")
        return vocab
    
    def encode(self, text: str, max_length: int = None, padding: bool = True, return_tensors: str = None):
        encoded = [self.token_to_id.get(char, self.token_to_id.get("<UNK>", 1)) for char in text]
        
        if padding:
            if max_length and len(encoded) < max_length:
                padding_length = max_length - len(encoded)
                encoded.extend([self.token_to_id[self.padding_token]] * padding_length)
            else:
                encoded = encoded[:max_length]
        
        token_type_ids = [0] * len(encoded)
        attention_mask = [1 if token != self.token_to_id[self.padding_token] else 0 for token in encoded]
        
        output = {
            "input_ids": encoded,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }
        
        if return_tensors == 'pt':
            output = {key: torch.tensor(value) for key, value in output.items()}
        
        return output
    
    def charword_encode(self, text: str, max_char_in_word: int = 12, max_word_length: int = None, padding: bool = True, return_tensors: str = None):
        batch_input_ids = []
        batch_char_lengths = []
        
        words = text.split('་')
        for word in words:
            word_ids = [self.token_to_id.get(char, self.token_to_id.get("<UNK>", -1)) for char in word]
            
            if len(word_ids) > max_char_in_word:
                word_ids = word_ids[:max_char_in_word]
            elif padding and len(word_ids) < max_char_in_word:
                word_ids += [self.token_to_id[self.padding_token]] * (max_char_in_word - len(word_ids))
            
            batch_input_ids.append(word_ids)
            batch_char_lengths.append(max(len(word), 1))
        
        if max_word_length and len(batch_input_ids) > max_word_length:
            batch_input_ids = batch_input_ids[:max_word_length]
            batch_char_lengths = batch_char_lengths[:max_word_length]
        elif padding and len(batch_input_ids) < max_word_length:
            padding_length = max_word_length - len(batch_input_ids)
            batch_input_ids += [[self.token_to_id[self.padding_token]] * max_char_in_word] * padding_length
            batch_char_lengths += [1] * padding_length
        
        batch_lengths = len(batch_input_ids)
        
        token_type_ids = [0] * len(batch_input_ids)
        attention_mask = [1] * len(batch_input_ids)
        #attention_mask = [1] * min(len(words), max_word_length) + [0] * (len(batch_input_ids) - min(len(words), max_word_length))

        output = {
            "input_ids": batch_input_ids,
            "batch_lengths": batch_lengths,
            "batch_char_lengths": batch_char_lengths,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask}
        
        if return_tensors == 'pt':
            output = {key: torch.tensor(value) for key, value in output.items()}
        
        return output
    
    def sc_vector(self, word):
        a = np.zeros(self.vocab_size, dtype=int)
        b = np.zeros(self.vocab_size, dtype=int)
        c = np.zeros(self.vocab_size, dtype=int)

        if word:
            a[self.token_to_id.get(word[0], self.token_to_id['<UNK>'])] = 1

            if len(word) >= 3:
                indices = [self.token_to_id.get(char, self.token_to_id['<UNK>']) for char in word[1:-1]]
                np.add.at(b, indices, 1)

            if len(word) >= 2:
                c[self.token_to_id.get(word[-1], self.token_to_id['<UNK>'])] = 1

        return np.concatenate((a, b, c))
    
    def sc_encode(self, text: str, max_length: int = None, padding: bool = True, return_tensors: str = None):
        words = text.split('་')
        
        batch_screps = [self.sc_vector(word) for word in words]
        vector_length = len(batch_screps[0])

        batch_screps = np.array(batch_screps, dtype=int)

        if max_length:
            if len(batch_screps) > max_length:
                batch_screps = batch_screps[:max_length]
            elif padding and len(batch_screps) < max_length:
                padding_screp = np.zeros(vector_length, dtype=int)
                padding_array = np.tile(padding_screp, (max_length - len(batch_screps), 1))
                batch_screps = np.vstack((batch_screps, padding_array))

        batch_screps = torch.tensor(batch_screps, dtype=torch.float32)
        
        token_type_ids = [0] * len(batch_screps)
        attention_mask = [1] * min(len(words), max_length) + [0] * (len(batch_screps) - min(len(words), max_length))

        output = {"input_ids": batch_screps, 
                  "batch_lengths": len(batch_screps),
                  "token_type_ids": token_type_ids,
                  "attention_mask": attention_mask}

        if return_tensors == 'pt':
            output = {key: torch.tensor(value) for key, value in output.items()}
        return output
    
    def decode(self, token_ids, seq=" "):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        decoded = []
        for token_id in token_ids:
            if token_id == self.token_to_id.get(self.padding_token):
                continue
            decoded.append(self.id_to_token.get(token_id, "<UNK>"))
            decoded.append(seq)
        return ''.join(decoded)
    
    def batch_decode(self, token_ids_batch, seq):
        decoded_batch = []
        
        for token_ids in token_ids_batch:
            decoded_text = self.decode(token_ids, seq)
            decoded_batch.append(decoded_text)
        
        return decoded_batch
    
    def add_token(self, token: str):
        if token not in self.token_to_id:
            new_id = self.vocab_size
            self.token_to_id[token] = new_id
            self.id_to_token[new_id] = token
            self.vocab.append(token)
            self.vocab_size += 1


class BPETokenizer:
    def __init__(self, vocab_file: str = None, vocab: list = None, padding_token: str = "<PAD>", special_tokens=None):
        if vocab_file:
            self.vocab = self.load_vocab(vocab_file)
        elif vocab:
            self.vocab = vocab
        else:
            raise ValueError("please provide file or list")

        if padding_token not in self.vocab:
            self.vocab.append(padding_token)
        
        self.special_tokens = special_tokens if special_tokens else ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.padding_token = padding_token
        self.vocab_size = len(self.vocab)
        self.token_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_token = {idx: char for idx, char in enumerate(self.vocab)}

        # Add special tokens to the vocab
        for token in self.special_tokens:
            if token not in self.token_to_id:
                self.add_token(token)

    def load_vocab(self, vocab_file: str) -> list:
        """从vocab.txt文件加载词汇表"""
        vocab = []
        try:
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"File {vocab_file} not found")
        except Exception as e:
            print(f"Error: {e}")
        return vocab

    def encode(self, text: str, max_length: int = None, padding: bool = True, return_tensors: str = None):
        """BPE编码"""
        # Step 1: Tokenize the text using BPE
        encoded = self.tokenize(text)

        # Step 2: Convert tokens to IDs
        encoded_ids = [self.token_to_id.get(token, self.token_to_id.get("<UNK>", 1)) for token in encoded]

        # Step 3: Apply padding
        if padding:
            if max_length and len(encoded_ids) < max_length:
                padding_length = max_length - len(encoded_ids)
                encoded_ids.extend([self.token_to_id[self.padding_token]] * padding_length)
            else:
                encoded_ids = encoded_ids[:max_length]

        token_type_ids = [0] * len(encoded_ids)
        attention_mask = [1 if token != self.token_to_id[self.padding_token] else 0 for token in encoded_ids]

        output = {
            "input_ids": encoded_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }

        if return_tensors == 'pt':
            output = {key: torch.tensor(value) for key, value in output.items()}

        return output

    def decode(self, token_ids, seq=" "):
        """解码token_ids为文本"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        decoded = []
        for token_id in token_ids:
            if token_id == self.token_to_id.get(self.padding_token):
                continue
            decoded.append(self.id_to_token.get(token_id, "<UNK>"))
            decoded.append(seq)
        return ''.join(decoded)

    def batch_decode(self, token_ids_batch, seq):
        """批量解码"""
        decoded_batch = []
        for token_ids in token_ids_batch:
            decoded_text = self.decode(token_ids, seq)
            decoded_batch.append(decoded_text)
        return decoded_batch

    def add_token(self, token: str):
        """为token添加ID映射"""
        if token not in self.token_to_id:
            new_id = self.vocab_size
            self.token_to_id[token] = new_id
            self.id_to_token[new_id] = token
            self.vocab.append(token)
            self.vocab_size += 1

    def tokenize(self, text: str):
        """BPE分词，依据字典进行分词"""
        # 1. 拆分词语为单个字符（字符级分词）
        text = ' '.join(list(text))  # 将每个字符用空格分开
        tokens = text.split()  # 字符级分词
        # 2. 对每个token进行BPE合并（假设token为子词或词）
        for i in range(len(tokens)):
            token = tokens[i]
            if token not in self.token_to_id:
                tokens[i] = "<UNK>"  # 用<UNK>表示不在词汇表中的子词
        return tokens