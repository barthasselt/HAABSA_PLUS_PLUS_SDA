import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import sentiwordnet as swn
import xml.etree.ElementTree as ET
import math
import os

import maskSelectionMethods
from maskSelectionMethods import mask_words_random, mask_words_pos, mask_words_senti

# Load the fine-tuned model and tokenizer
model_path = '/content/drive/MyDrive/Thesis/DataAugmentation/models/fine_tuned_CBERT2016_model'
tokenizer_path = '/content/drive/MyDrive/Thesis/DataAugmentation/models/fine_tuned_CBERT2016_tokenizer'

# Define the path to the text file and augmented data path
txt_file = '/content/drive/MyDrive/Thesis/DataAugmentation/data/processedData/data_16_train.txt'
augmented_data_path = '/content/drive/MyDrive/Thesis/DataAugmentation/augmentedData/2016'
os.makedirs(augmented_data_path, exist_ok=True)

# Define a custom BERT model with resized token type embeddings
class CustomBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.resize_token_type_embeddings(3)

    def resize_token_type_embeddings(self, new_num_types):
        old_embeddings = self.bert.embeddings.token_type_embeddings
        new_embeddings = torch.nn.Embedding(new_num_types, old_embeddings.embedding_dim)
        new_embeddings.to(old_embeddings.weight.device)
        if new_num_types > old_embeddings.num_embeddings:
            new_embeddings.weight.data[:old_embeddings.num_embeddings] = old_embeddings.weight.data
        else:
            new_embeddings.weight.data = old_embeddings.weight.data[:new_num_types]
        self.bert.embeddings.token_type_embeddings = new_embeddings

# Load the custom model with ignore_mismatched_sizes=True
model = CustomBertForMaskedLM.from_pretrained(model_path, ignore_mismatched_sizes=True)

# Set the model to evaluation mode
model.eval()

tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# Function to parse the input text file
def parse_txt(file):
    try:
        with open(file, 'r') as f:
            lines = f.readlines()
        
        sentences = []
        for i in range(0, len(lines), 3):
            text = lines[i].strip()
            target = lines[i+1].strip()
            polarity = lines[i+2].strip()
            sentences.append((text, target, polarity))
        
        return sentences
    except Exception as e:
        print(f"Error reading text file: {e}")
        return []

# Extract sentences from the text file
sentences = parse_txt(txt_file)

if not sentences:
    print("No sentences found in the text file.")
    exit()

def fill_masked_words(mask_indices, words, polarity, model, tokenizer, top_k=2):
    # Strip any whitespace and convert to integer
    try:
        polarity = int(str(polarity).strip())
    except ValueError:
        raise ValueError(f"Polarity value '{polarity}' is not a valid integer")

    # Convert polarity from -1, 0, 1 to 0, 1, 2
    polarity_mapping = {-1: 0, 0: 1, 1: 2}
    if polarity not in polarity_mapping:
        raise ValueError("Polarity must be -1, 0, or 1")

    mapped_polarity = polarity_mapping[polarity]

    original_words = [words[idx] for idx in mask_indices]

    for idx in mask_indices:
        words[idx] = tokenizer.mask_token
    masked_text = ' '.join(words)

    # Debug: Print the masked text
    print(f"Masked text: {masked_text}")

    inputs = tokenizer(masked_text, return_tensors='pt')
    input_ids = inputs.input_ids

    # Debug: Print input IDs
    print(f"Input IDs: {input_ids}")

    token_type_ids = torch.tensor([mapped_polarity] * len(input_ids[0])).unsqueeze(0)

    # Debug: Print token type IDs
    print(f"Token Type IDs: {token_type_ids}")

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=token_type_ids)
        predictions = outputs.logits

    new_words = []
    mask_token_indices = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].tolist()

    for mask_token_index, original_word in zip(mask_token_indices, original_words):
        token_logits = predictions[0, mask_token_index, :]
        top_k_ids = torch.topk(token_logits, top_k).indices

        # Debug: Print the top-k token IDs for each masked position
        print(f"Top-k token IDs for mask at position {mask_token_index}: {top_k_ids}")

        # Select a replacement that is not the original word
        replacement_word = None
        for token_id in top_k_ids:
            candidate_word = tokenizer.decode([token_id.item()], skip_special_tokens=True)
            if candidate_word != original_word:
                replacement_word = candidate_word
                break
        if not replacement_word:
            replacement_word = tokenizer.decode([top_k_ids[0].item()], skip_special_tokens=True)

        new_words.append(replacement_word)
        words[mask_indices[mask_token_indices.index(mask_token_index)]] = replacement_word

    # Join words correctly
    augmented_text = tokenizer.convert_tokens_to_string(words)

    # Debug: Print the final augmented text
    print(f"Augmented text: {augmented_text}")

    return augmented_text


# Function to prepare and save the augmented data
def prepare_and_save_augmented_data(augmentation_method, postag=None):
    augmented_sentences = []

    for text_with_placeholder, target, polarity in sentences:
        # Original sentence with the placeholder
        augmented_sentences.append((text_with_placeholder, target, polarity))
        
        # Mask words based on selected method
        if augmentation_method == 'Random':
            mask_indices, words = mask_words_random(text_with_placeholder)
        elif augmentation_method == 'POS':
            mask_indices, words = mask_words_pos(text_with_placeholder, postag)
        elif augmentation_method == 'Sentiment':
            mask_indices, words = mask_words_senti(text_with_placeholder)
        else:
            raise ValueError("Invalid augmentation method selected. Choose from 'Random', 'POS', 'Sentiment'.")

        if mask_indices:
          # Replace $T$ with the target word in the list of words
          words = [target if word == '$T$' else word for word in words]

          # Fill masked words
          augmented_text = fill_masked_words(mask_indices, words, polarity, model, tokenizer)

          # Find and replace the target with $T$
          augmented_with_placeholder = augmented_text.replace(target, '$T$', 1)

          augmented_sentences.append((augmented_with_placeholder, target, polarity))

    if augmentation_method == 'POS':
        augmented_txt_path = os.path.join(augmented_data_path, f'CBERT2016augmented_data_{augmentation_method}_{postag}.txt')
    else:
        augmented_txt_path = os.path.join(augmented_data_path, f'CBERT2016augmented_data_{augmentation_method}.txt')

    with open(augmented_txt_path, 'w') as f:
        for text, target, polarity in augmented_sentences:
            f.write(f"{text}\n{target}\n{polarity}\n")

    print(f"Augmented dataset successfully saved to {augmented_txt_path}")

# Create augmented data files for different methods
prepare_and_save_augmented_data('Random')
prepare_and_save_augmented_data('Sentiment')
for pos_tag in ['VB', 'JJ', 'NN', 'RB', 'Combi']:
    prepare_and_save_augmented_data('POS', postag=pos_tag)
