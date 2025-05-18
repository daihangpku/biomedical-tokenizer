from transformers import BertTokenizer
import json
import pandas as pd
from transformers import BertTokenizerFast

hoc_dataset_path = "data/HoC/train.parquet"
hoc_data = pd.read_parquet(hoc_dataset_path)


original_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
expanded_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
with open("data/extracted_token.json", "r") as f:
    new_tokens = json.load(f)
print(f"Adding {len(new_tokens)} new tokens to the tokenizer.")
expanded_tokenizer.add_tokens(new_tokens)
print(expanded_tokenizer.get_vocab())


sample_sentences = hoc_data["text"].sample(3).tolist()

for i, sentence in enumerate(sample_sentences):
    print(f"Sentence {i + 1}: {sentence}")
    
    
    original_tokens = original_tokenizer.tokenize(sentence)
    print(f"Original BERT Tokens: {original_tokens}")
    
    
    expanded_tokens = expanded_tokenizer.tokenize(sentence)
    print(f"Expanded BERT Tokens: {expanded_tokens}")
    
    
    print(f"Difference in token count: {len(expanded_tokens) - len(original_tokens)}")
    print("-" * 50)


original_lengths = []
expanded_lengths = []

for sentence in hoc_data["text"]:

    original_tokens = original_tokenizer.tokenize(sentence)
    original_lengths.append(len(original_tokens))
    
    expanded_tokens = expanded_tokenizer.tokenize(sentence)
    expanded_lengths.append(len(expanded_tokens))


original_avg_length = sum(original_lengths) / len(original_lengths)
expanded_avg_length = sum(expanded_lengths) / len(expanded_lengths)

print(f"Average token length (Original BERT): {original_avg_length:.2f}")
print(f"Average token length (Expanded BERT): {expanded_avg_length:.2f}")
print(f"Difference in average length: {expanded_avg_length - original_avg_length:.2f}")