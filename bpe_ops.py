import re, collections
from typing import Dict, Tuple

# This script implements the Byte Pair Encoding (BPE) algorithm to iteratively 
# merge the most frequent adjacent character pairs into subword tokens.

def get_stats(vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """
    Identifies all adjacent symbol pairs in the vocabulary and counts their collective frequencies.
    """
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs


def merge_vocab(pair: Tuple[str, str], v_in: Dict[str, int]) -> Dict[str, int]:
    """
    Replaces all occurrences of a specific symbol pair with a single merged token across the vocabulary.
    """
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


# Initial tokenized vocabulary with frequency counts
vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}

num_merges = 10

# Iterative process to extract the best pair and update the vocabulary
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)