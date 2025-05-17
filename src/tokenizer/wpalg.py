from transformers import AutoTokenizer
from collections import defaultdict


def wordpiece(training_corpus, vocab_size):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    word_freqs = defaultdict(int)
    for text in training_corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1

    alphabet = []
    for word in word_freqs.keys():
        if word[0] not in alphabet:
            alphabet.append(word[0])
        for letter in word[1:]:
            if f"##{letter}" not in alphabet:
                alphabet.append(f"##{letter}")

    alphabet.sort()

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()

    # Do NOT add your above this line.
    #======
    
    def tokenize_word(word):
        if word in vocab:                   # already an atomic piece
            return [word]
        tokens = []
        i = 0
        while i < len(word):
            j = len(word)
            cur_substr = None
            while i < j:
                substr = word[i:j]
                if i > 0:
                    substr = "##" + substr    # continuation mark
                if substr in vocab:
                    cur_substr = substr
                    break
                j -= 1
            if cur_substr is None:           # fall back to [UNK]
                return ["[UNK]"]
            tokens.append(cur_substr)
            i += len(cur_substr.replace("##", ""))
        return tokens

    def get_stats():
        """Return pair-frequency map of current segmentation."""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            tokens = tokenize_word(word)
            for a, b in zip(tokens, tokens[1:]):
                pairs[(a, b)] += freq
        return pairs

    def merge_pair(best_pair):
        """Replace occurrences of best_pair with merged token."""
        a, b = best_pair
        new_token = a + b.lstrip("##") if b.startswith("##") else a + b
        if not new_token.startswith("##"):
            new_token = new_token           # first piece already correct
        vocab.append(new_token)
        # No need to update word_freqs; the tokenizer function uses vocab live.
    # from tqdm import tqdm 
    # pbar = tqdm(total=vocab_size - len(vocab), desc="Building vocab")
    while len(vocab) < vocab_size:
        stats = get_stats()
        if not stats:
            break
        best = max(stats, key=stats.get)
        merge_pair(best)
    #     pbar.update(1)
    # pbar.close()
    #======
    # Do NOT add your below this line.

    return vocab

if __name__ == "__main__":
    default_training_corpus = [
        "peking university is located in haidian district",
        "computer science is the flagship major of peking university",
        "the school of electronic engineering and computer science enrolls approximately five hundred new students each year"  
    ]

    default_vocab_size = 120

    my_vocab = wordpiece(default_training_corpus, default_vocab_size)

    print('The vocab:', my_vocab)

    def encode_word(custom_vocab, word):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in custom_vocab:
                i -= 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens

    def tokenize(custom_vocab, text):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        encoded_words = [encode_word(custom_vocab, word) for word in pre_tokenized_text]
        return sum(encoded_words, [])

    print('Tokenization result:', tokenize(my_vocab, 'nous etudions a l universite de pekin'))
