import torch
from transformers import BertTokenizerFast, BertForMaskedLM
import random
import os
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import sentiwordnet as swn

import maskSelectionMethods
from maskSelectionMethods import mask_words_random, mask_words_pos, mask_words_senti

# Define paths
txt_file = '/content/drive/MyDrive/Thesis/DataAugmentation/data/processedData/data_16_train.txt'
augmented_data_path = '/content/drive/MyDrive/Thesis/DataAugmentation/augmentedData/2016'

# Ensure the directory for augmented data exists
os.makedirs(augmented_data_path, exist_ok=True)

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

# Function to fill masked words
def fill_masked_words(mask_indices, words, model, tokenizer, top_k=5):
    original_words = [words[idx] for idx in mask_indices]
    masked_words = words[:]
    
    for idx in mask_indices:
        masked_words[idx] = '[MASK]'
    
    augmented_text = ' '.join(masked_words)
    inputs = tokenizer(augmented_text, return_tensors='pt')
    input_ids = inputs.input_ids

    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits

    new_words = []
    mask_token_indices = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].tolist()
    
    for mask_token_index, original_word in zip(mask_token_indices, original_words):
        token_logits = predictions[0, mask_token_index, :]
        top_k_ids = torch.topk(token_logits, top_k).indices
        
        # Select a replacement that is not the original word
        replacement_word = None
        for token_id in top_k_ids:
            candidate_word = tokenizer.decode([token_id], skip_special_tokens=True)
            if candidate_word != original_word:
                replacement_word = candidate_word
                break
        if not replacement_word:
            replacement_word = tokenizer.decode([top_k_ids[0]], skip_special_tokens=True)
        
        new_words.append(replacement_word)
        masked_words[mask_indices[mask_token_indices.index(mask_token_index)]] = replacement_word

    final_augmented_text = ' '.join(masked_words)
    
    # Print statements to show the augmentation process
    print(f"Original sentence: {' '.join(words)}")
    print(f"Masked words: {original_words}")
    print(f"Generated replacement words: {new_words}")
    print(f"New augmented sentence: {final_augmented_text}")
    
    return final_augmented_text

# Function to prepare and save the augmented data
def prepare_and_save_augmented_data(method_type, augmentation_method, postag=None):
    augmented_sentences = []

    if method_type == 'BERT':
      model_path = './models/fine_tuned_BERT_2016_model'
      tokenizer_path = './models/fine_tuned_BERT_2016_tokenizer'

    # Load the fine-tuned model and tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    model = BertForMaskedLM.from_pretrained(model_path)

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
            augmented_text = fill_masked_words(mask_indices, words, model, tokenizer)

            # Place $T$ back into the augmented text
            augmented_words = augmented_text.split()
            
            augmented_with_placeholder = ' '.join(augmented_words).replace(target, '$T$', 1)

            augmented_sentences.append((augmented_with_placeholder, target, polarity))

    if augmentation_method == 'POS':
        augmented_txt_path = os.path.join(augmented_data_path, f'{method_type}2016augmented_data_{augmentation_method}_{postag}.txt')
    else:
        augmented_txt_path = os.path.join(augmented_data_path, f'{method_type}2016augmented_data_{augmentation_method}.txt')

    with open(augmented_txt_path, 'w') as f:
        for text, target, polarity in augmented_sentences:
            f.write(f"{text}\n{target}\n{polarity}\n")

    print(f"Augmented dataset successfully saved to {augmented_txt_path}")

# Create augmented data files for different methods
for method in [ 'BERT']:
    prepare_and_save_augmented_data(method, 'Random')
    prepare_and_save_augmented_data(method, 'Sentiment')
    for pos_tag in ['VB', 'JJ', 'NN', 'RB', 'Combi']:
        prepare_and_save_augmented_data(method, 'POS', postag=pos_tag)
