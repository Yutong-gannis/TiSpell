from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from option import parse_args
import torch
from model.cnnlstm import CharCNNWordLSTMModel
from model.lstmlstm import CharLSTMWordLSTMModel
from model.sclstm import SCLSTM
from model.bertcharword import BertCharWord
from model.roberta_bilstm import Roberta_BiLSTM
from model.tispell_roberta import TiSpell_RoBERTa
from model.tispell_cino import TiSpell_CINO
from model.soft_masked_roberta import SoftMaskedRoBERTa

args = parse_args()

print("cnn_lstm: ")
cnn_lstm = CharCNNWordLSTMModel(args)
print(sum(p.numel() for p in cnn_lstm.parameters()) / 1e6)

print("lstm_lstm: ")
lstm_lstm = CharLSTMWordLSTMModel(args)
print(sum(p.numel() for p in lstm_lstm.parameters()) / 1e6)

print("sc_lstm: ")
sc_lstm = SCLSTM(args)
print(sum(p.numel() for p in sc_lstm.parameters()) / 1e6)

print("bert_char: ")
bert = BertCharWord(args)
print(sum(p.numel() for p in bert.parameters()) / 1e6)

tokenizer = AutoTokenizer.from_pretrained("tokenizer/tibetan_roberta", use_fast=False)

print("t5: ")
model = AutoModelForSeq2SeqLM.from_pretrained("tokenizer/t5_large")
model.resize_token_embeddings(len(tokenizer))
print(sum(p.numel() for p in model.parameters()) / 1e6)

print("roberta: ")
roberta = AutoModelForCausalLM.from_pretrained("./tokenizer/tibetan_roberta")
roberta.resize_token_embeddings(len(tokenizer))
print(sum(p.numel() for p in roberta.parameters()) / 1e6)

print("bert: ")
bert = AutoModelForCausalLM.from_pretrained("./tokenizer/tibetan_bert")
bert.resize_token_embeddings(len(tokenizer))
print(sum(p.numel() for p in bert.parameters()) / 1e6)

print("cino: ")
cino = AutoModelForCausalLM.from_pretrained("./tokenizer/cino_base_v2")
cino.resize_token_embeddings(len(tokenizer))
print(sum(p.numel() for p in cino.parameters()) / 1e6)

print("roberta_bilstm: ")
model = Roberta_BiLSTM("./tokenizer/tibetan_roberta", tokenizer)
print(sum(p.numel() for p in model.parameters()) / 1e6)

print("soft_masked_roberta: ")
model = SoftMaskedRoBERTa("./tokenizer/tibetan_roberta", tokenizer, torch.device("cpu"))
print(sum(p.numel() for p in model.parameters()) / 1e6)

print("tispell_roberta: ")
model = TiSpell_RoBERTa("./tokenizer/tibetan_roberta", tokenizer)
print(sum(p.numel() for p in model.parameters()) / 1e6)

print("tispell_cino: ")
model = TiSpell_CINO("./tokenizer/tibetan_roberta", tokenizer)
print(sum(p.numel() for p in model.parameters()) / 1e6)
