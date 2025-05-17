from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW 
import torch
import json
import os
from tqdm import tqdm
from evaluate import evaluate_model, calculate_metrics
from dataset import HoCDataset
import matplotlib.pyplot as plt
def train_classifier(config, model, train_dataset, test_dataset=None):
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["num_epochs"])
    lr = float(config["training"]["learning_rate"])
    test_epoch = int(config["training"]["test_epoch"])
    save_epoch = int(config["training"]["save_epoch"])
    model_saving_dir = config["training"]["model_saving_dir"]
    log_dir = config["training"]["log_dir"]
    os.makedirs(model_saving_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Training with batch size: {batch_size}, epochs: {epochs}, learning rate: {lr}")


    train_losses = []
    test_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Train_Loss: {avg_loss:.4f}")
        if test_dataset and (epoch + 1) % test_epoch == 0:
            preds, labels = evaluate_model(model, test_dataset, config)
            accuracy, precision, recall, f1 = calculate_metrics(preds, labels)
            test_metrics["accuracy"].append(accuracy)
            test_metrics["precision"].append(precision)
            test_metrics["recall"].append(recall)
            test_metrics["f1"].append(f1)
            print(f'Accuracy: {accuracy:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1:.4f}')
        if (epoch + 1) % save_epoch == 0:
            model.save_pretrained(f"{model_saving_dir}/{epoch + 1}")
            print(f"Model saved at epoch {epoch + 1}")
    save_training_logs(train_losses, test_metrics, log_dir)

def save_training_logs(train_losses, test_metrics, log_dir):

    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/train_losses.json", "w") as f:
        json.dump(train_losses, f, indent=4)
    with open(f"{log_dir}/test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    # 绘制训练损失图
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(f"{log_dir}/train_loss.png")
    plt.close()

    for metric, values in test_metrics.items():
        plt.figure()
        plt.plot(range(1, len(values) + 1), values, label=metric.capitalize())
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.title(f"Test {metric.capitalize()}")
        plt.legend()
        plt.savefig(f"{log_dir}/test_{metric}.png")
        plt.close()
        
if __name__ == "__main__":
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description="Train a WordPiece tokenizer and extract domain-specific tokens.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()
    # Load configuration
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=11)
    with open(config["paths"]["extracted_token"], "r") as f:
            new_tokens = json.load(f)
    print(f"Adding {len(new_tokens)} new tokens to the tokenizer.")
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    train_dataset = HoCDataset(config["paths"]["hoc_train_dataset"], tokenizer)
    test_dataset = HoCDataset(config["paths"]["hoc_test_dataset"], tokenizer)
    # labels = [item['label'] for item in train_dataset.data]
    # print(f"Labels range: min={min(labels)}, max={max(labels)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_classifier(config, model, train_dataset, test_dataset)