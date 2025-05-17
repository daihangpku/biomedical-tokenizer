from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
import json
from dataset import HoCDataset
def evaluate_model(model, dataset, config):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

def calculate_metrics(preds, labels):
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return accuracy, precision, recall, f1
    

if __name__ == "__main__":
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description="Train a WordPiece tokenizer and extract domain-specific tokens.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()
    # Load configuration
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=11)
    with open(config["paths"]["extracted_token"], "r") as f:
            new_tokens = json.load(f)
    print(f"Adding {len(new_tokens)} new tokens to the tokenizer.")
    tokenizer.add_tokens(new_tokens)

    model.resize_token_embeddings(len(tokenizer))
    test_dataset = HoCDataset(config["paths"]["hoc_test_dataset"], tokenizer)

    model.to(device)
    # Load model checkpoint
    checkpoint_path = config["evaluation"]["model_dir"]
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded model checkpoint from {checkpoint_path}")
    
    preds, labels = evaluate_model(model, test_dataset, device)
    accuracy, precision, recall, f1 = calculate_metrics(preds, labels)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')