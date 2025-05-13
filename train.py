import re
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AdamW, get_scheduler, AutoTokenizer
from tqdm import tqdm
from option import parse_args
from dataloader.dataset import TibetanDatasetOneTokenizer
from metrics import compute_precision_recall_f1
from plot import plot_results
from model.tispell_roberta import TiSpell_RoBERTa

def train(args, model, train_loader, criterion, optimizer, lr_scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        tokenized_text = batch
        correct_input_ids = tokenized_text['correct']['input_ids'].to(device)
        corrupt_input_ids = tokenized_text['random_corrupt']['input_ids'].to(device)
        corrupt_attention_mask = tokenized_text['random_corrupt']['attention_mask'].to(device)
        correct_tag_ids = tokenized_text['mask']['input_ids'].to(device)
        logit, logit_c = model(corrupt_input_ids, attention_mask=corrupt_attention_mask)
        
        loss = criterion(logit.permute(0, 2, 1), correct_input_ids)
        loss_c = criterion(logit_c.permute(0, 2, 1), correct_tag_ids)
        loss = loss + args.w_c * loss_c
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validation(model, test_loader, tokenizer, device, epoch, results_per_epoch):
    model.eval()
    results = {key: {'pre': 0, 'rec': 0, 'f1': 0, 'count': 0} for key in [
        'correct', 'random_corrupt', 'char_random_delete', 'char_random_insert',
        'char_tall_short_replace', 'char_homomorphic_replace', 'char_inner_syllable_exchange',
        'char_near_syllable_exchange', 'syllable_random_delete', 'syllable_random_exchange', 'syllable_random_merge'
    ]}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            tokenized_text = batch
            correct_input_ids = tokenized_text['correct']['input_ids'].to(device)
            correct_texts = tokenizer.batch_decode(correct_input_ids, seq='་', skip_special_tokens=True)
            
            for key in tokenized_text.keys():
                corrupt_input_ids = tokenized_text[key]['input_ids'].to(device)
                corrupt_attention_mask = tokenized_text[key]['attention_mask'].to(device)

                logit, _ = model(corrupt_input_ids, attention_mask=corrupt_attention_mask)
                pred_input_ids = torch.argmax(logit, dim=-1)
                pred_texts = tokenizer.batch_decode(pred_input_ids, seq='་', skip_special_tokens=True)

                for pred_text, correct_text in zip(pred_texts, correct_texts):
                    pred_text = pred_text.replace('་', ' ')
                    correct_text = correct_text.replace('་', ' ')
                    precision, recall, f1 = compute_precision_recall_f1(re.split(' ', pred_text), re.split(' ', correct_text))

                    results[key]['pre'] += precision
                    results[key]['rec'] += recall
                    results[key]['f1'] += f1
                    results[key]['count'] += 1

    for corruption_type, metrics in results.items():
        if metrics['count'] > 0:
            metrics['pre'] /= metrics['count']
            metrics['rec'] /= metrics['count']
            metrics['f1'] /= metrics['count']
            print(f"Type: {corruption_type} | pre: {metrics['pre']:.4f} | rec: {metrics['rec']:.4f} | f1: {metrics['f1']:.4f}")
        else:
            metrics['pre'] = metrics['rec'] = metrics['f1'] = 0

    results_per_epoch[epoch] = results
    for key, metrics in results.items():
        for metric_name, value in metrics.items():
            if metric_name != 'count':
                wandb.log({f"{key}_{metric_name}": value})
            
    model.roberta.save_pretrained("weights/tispell_roberta")
    tokenizer.save_pretrained("weights/tispell_roberta")
    torch.save(model.state_dict(), "weights/tispell_roberta/tispell_roberta.pth")
    return results_per_epoch

def main():
    args = parse_args()
    
    wandb.login(key="1cbc332e3014e365955e98644476031bf8964c18")
    wandb.init(
        project="TiSpell",
        name="tispell_roberta",

        config={
        "model": "tispell_roberta",
        "language": "Tibetan",
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_sample": args.num_sample,
        "w_c": args.w_c
        }
    )
    
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained("./pretrained_models/tibetan_roberta", use_fast=False)
    model = TiSpell_RoBERTa("./pretrained_models/tibetan_roberta", tokenizer)
    model.to(device)

    dataset = TibetanDatasetOneTokenizer(args, tokenizer)
    train_size = int(args.split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = len(train_loader) * args.epochs
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    results_per_epoch = {}
    for epoch in range(args.epochs):
        train_dataset.dataset.mode = 'train'
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss = train(args, model, train_loader, criterion, optimizer, lr_scheduler, device)
        print(f"Training loss: {train_loss:.4f}")
        test_dataset.dataset.mode = 'test'
        results_per_epoch = validation(model, test_loader, tokenizer, device, epoch, results_per_epoch)
    plot_results(range(args.epochs), results_per_epoch, results_per_epoch[0].keys(), metric="pre")
    plot_results(range(args.epochs), results_per_epoch, results_per_epoch[0].keys(), metric="rec")
    plot_results(range(args.epochs), results_per_epoch, results_per_epoch[0].keys(), metric="f1")

if __name__ == "__main__":
    main()
