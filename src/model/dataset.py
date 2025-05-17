from torch.utils.data import Dataset
import torch
import pandas as pd
class HoCDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        df = pd.read_parquet(file_path)
        
        return [{"text": row["text"], "label": row["label"]} for _, row in df.iterrows()]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }