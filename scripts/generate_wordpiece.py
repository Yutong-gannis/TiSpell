from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# 初始化 WordPiece 模型
tokenizer = Tokenizer(models.WordPiece())

# 设置预分词器，通常使用 'BertPreTokenizer'
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

# 设置解码器
tokenizer.decoder = decoders.WordPiece()

# 准备训练器
trainer = trainers.WordPieceTrainer(
    vocab_size=32000, 
    min_frequency=2,  # 最小频率设置
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

# 训练模型
files = ["tibetan_news_classification.txt"]  # 用你的数据集路径替换
tokenizer.train(files, trainer)

# 保存模型
tokenizer.save("wordpiece_tibetan.json")