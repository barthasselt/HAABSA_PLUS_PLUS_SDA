import random
import os
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import sentiwordnet as swn

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
nltk.download('wordnet')

# 1. Function to mask words randomly without masking target phrases and masking at most 15% of the real words
def mask_words_random(text):
    words = text.split()
    num_words = len(words)

    # Find indices of the $T$ placeholder
    target_indices = set()
    for i, word in enumerate(words):
        if word == '$T$':
            target_indices.add(i)

    # Identify maskable indices
    maskable_indices = [i for i in range(num_words) if i not in target_indices and words[i].isalpha() and i != 0]
    num_real_words = len(maskable_indices) + len(target_indices)
    max_num_to_mask = max(1, math.floor(num_real_words * 0.15))

    if num_real_words < 1:
        print(f"Skipping text (not enough words to mask): {text}")
        return [], words

    # Ensure we do not mask more words than available
    if len(maskable_indices) < max_num_to_mask:
        max_num_to_mask = len(maskable_indices)

    random.seed(530)
    mask_indices = random.sample(maskable_indices, max_num_to_mask)
    return mask_indices, words


# 2. Function to mask words based on POS tags without masking target phrases
def mask_words_pos(text, POSTAG):
    words = text.split()
    num_words = len(words)    
    pos_tags = pos_tag(words)
    
    # Define all variants of specified POS tags
    pos_tag_variants = {
        'VB': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'JJ': ['JJ', 'JJR', 'JJS'],
        'NN': ['NN', 'NNS', 'NNP', 'NNPS'],
        'RB': ['RB', 'RBR', 'RBS']
    }

    # Determine the extended POS tags to mask
    if POSTAG == 'Combi':
      extended_pos_tags_to_mask = []
      for tags in pos_tag_variants.values():
          extended_pos_tags_to_mask.extend(tags)
    else:
      extended_pos_tags_to_mask = pos_tag_variants.get(POSTAG, [POSTAG])

    # Debug: print extended POS tags to mask
    print(f"Extended POS tags to mask: {extended_pos_tags_to_mask}")

    # Find indices of the $T$ placeholder
    target_indices = set()
    for i, word in enumerate(words):
        if word == '$T$':
            target_indices.add(i)

    # Identify maskable indices based on POS tags, excluding target phrases and punctuation and first word
    pos_indices = [i for i, tag in enumerate(pos_tags) if tag[1] in extended_pos_tags_to_mask and i not in target_indices and i != 0 and words[i].isalpha()]

    # Exclude punctuation from the total words count
    num_real_words = len([word for word in words if word.isalpha()])
    max_num_to_mask = max(1, math.floor(num_real_words * 0.15))

    # Debug: print POS tags and selected POS indices
    print(f"POS tags for sentence: {pos_tags}")
    print(f"Selected POS indices for masking: {pos_indices}")

    if len(pos_indices) < 1:
        print(f"Skipping text (not enough POS-tagged words to mask): {text}")
        return [], words

    num_to_mask = min(max_num_to_mask, len(pos_indices))

    random.seed(530)
    mask_indices = random.sample(pos_indices, num_to_mask)
    
    # Debug: print final mask indices
    print(f"Final mask indices: {mask_indices}")

    return mask_indices, words

def get_sentiwordnet_score(word):
    synsets = list(swn.senti_synsets(word))
    if not synsets:  # Punctuation get score 0
        return 0
    pos_score = synsets[0].pos_score()
    neg_score = synsets[0].neg_score()
    return max(pos_score, neg_score)

def mask_words_senti(text):
    words = text.split()
    num_words = len(words)        
    sentiment_scores = [get_sentiwordnet_score(word) for word in words]

    # Debug: print words and their sentiment scores
    print(f"Words and their sentiment scores: {list(zip(words, sentiment_scores))}")

    # Find indices of the $T$ placeholder
    target_indices = set()
    for i, word in enumerate(words):
        if word == '$T$':
            target_indices.add(i)

    # Identify maskable indices based on sentiment scores, excluding target phrases and punctuation
    sentiment_indices = [i for i, score in enumerate(sentiment_scores) if abs(score) > 0.0 and i not in target_indices and words[i].isalpha() and i != 0]

    # Debug: print selected sentiment indices
    print(f"Selected sentiment indices for masking: {sentiment_indices}")
    
    num_real_words = len([word for word in words if word.isalpha()])
    max_num_to_mask = max(1, math.floor(num_real_words * 0.15))

    if num_real_words < 1:
        print(f"Skipping text (not enough sentiment words to mask): {text}")
        return [], words

    # Ensure we do not mask more words than available
    num_to_mask = min(max_num_to_mask, len(sentiment_indices))

    random.seed(530)
    mask_indices = random.sample(sentiment_indices, num_to_mask)

    # Debug: print final mask indices
    print(f"Final mask indices: {mask_indices}")

    return mask_indices, words

