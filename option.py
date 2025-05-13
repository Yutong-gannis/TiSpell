import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a BERT model for Tibetan text classification")
    parser.add_argument("--data_path", type=str, default="./dataset/tibetan_news_classification", help="Path to dataset")
    parser.add_argument("--num_sample", type=int, default=500000, help="Number of samples for experiment")
    parser.add_argument("--split_ratio", type=float, default=0.999, help="Split ratio of train subset, valid subset")
    
    parser.add_argument("--max_char_length", type=int, default=256, help="Max length of character")
    parser.add_argument("--max_subword_length", type=int, default=96, help="Max length of subword")
    parser.add_argument("--max_word_length", type=int, default=64, help="Max length of word")
    parser.add_argument("--max_char_in_word", type=int, default=12, help="Max length of character in a word")
    
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--num_epochs_per_evaluation", type=int, default=1, help="Number of epochs per evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--mix", type=bool, default="False", help="If use mixture for training")
    parser.add_argument("--w_c", type=float, default="1.0", help="weight of character-level loss")
    return parser.parse_args()