import sentencepiece as spm
 
spm.SentencePieceTrainer.train('--input=tibetan_news_classification.txt --model_prefix=bpe_tibetan --vocab_size=32000 --character_coverage=0.9995 --model_type=bpe')