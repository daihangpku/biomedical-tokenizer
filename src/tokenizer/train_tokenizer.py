from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import json
from transformers import BertTokenizer
def train_wordpiece_tokenizer(config):
    # Initialize a WordPiece tokenizer
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    
    # Set pre-tokenizer and decoder
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece()

    # Initialize a trainer for the tokenizer
    trainer = trainers.WordPieceTrainer(
        vocab_size=config["tokenizer"]["vocab_coarse_size"], 
        min_frequency=config["tokenizer"]["min_frequency"],
        special_tokens=config["tokenizer"]["special_tokens"])
    
    # Load the corpus
    with open(config["paths"]["pubmed_corpus"], 'r') as f:
        corpus = [line.strip().lower() for line in f.readlines()] # Convert to lowercase

    # Train the tokenizer
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    # Save the trained tokenizer
    tokenizer.save(config["paths"]["full_token"])

def extract_domain_specific_tokens(config):
    # Load the trained tokenizer's vocabulary
    with open(config["paths"]["full_token"], 'r') as f:
        trained_tokenizer_data = json.load(f)
    trained_vocab = trained_tokenizer_data["model"]["vocab"]

    # Load the BERT tokenizer's vocabulary
    bert_tokenizer = BertTokenizer.from_pretrained(config["paths"]["bert_model"])
    bert_vocab = set(bert_tokenizer.vocab.keys())

    # Filter out tokens already in BERT's vocabulary
    domain_specific_tokens = [token for token in trained_vocab if token not in bert_vocab]

    # Sort tokens by frequency (if available) or alphabetically
    sorted_tokens = sorted(domain_specific_tokens, key=lambda token: trained_vocab[token])

    # Select the top `num_tokens` tokens
    top_tokens = sorted_tokens[:config["tokenizer"]["vocab_size"]]

    # Save the top tokens to a file
    with open(config["paths"]["extracted_token"], 'w') as f:
        json.dump(top_tokens, f, indent=4)

if __name__ == "__main__":
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description="Train a WordPiece tokenizer and extract domain-specific tokens.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()
    # Load configuration
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    train_wordpiece_tokenizer(config)

    extract_domain_specific_tokens(config)