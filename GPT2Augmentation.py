import random
import nltk
import torch
from nltk import pos_tag, word_tokenize
from nltk.corpus import sentiwordnet as swn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

import maskSelectionMethods
from maskSelectionMethods import mask_words_random, mask_words_pos, mask_words_senti

# Paths
txt_file = '/content/drive/MyDrive/Thesis/DataAugmentation/data/processedData/data_16_train.txt'
augmented_data_path = '/content/drive/MyDrive/Thesis/DataAugmentation/augmentedData/2016'
os.makedirs(augmented_data_path, exist_ok=True)

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
            from_index = text.find(target)
            to_index = from_index + len(target)
            text_with_placeholder = text.replace(target, '$T$')
            sentences.append((text_with_placeholder, target, polarity, from_index, to_index))
        
        return sentences
    except Exception as e:
        print(f"Error reading text file: {e}")
        return []

# Extract sentences from the text file
sentences = parse_txt(txt_file)

if not sentences:
    print("No sentences found in the text file.")
    exit()

def fill_masked_words(mask_indices, words, model, tokenizer, top_k=5):
    original_words = [words[idx] for idx in mask_indices]
    masked_words = words[:]
    replacement_words = []

    for idx in mask_indices:
        # Construct the sentence up to the word before the masked word
        left_part = words[:idx]
        left_text = ' '.join(left_part)

        # Encode the input text
        inputs = tokenizer(left_text, return_tensors='pt')

        # Generate the next word prediction with top_k sampling
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=len(inputs.input_ids[0]) + 1,
                num_return_sequences=top_k,
                do_sample=True,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=inputs.attention_mask
            )

        # Decode the generated tokens and extract potential next words
        generated_words = [tokenizer.decode(output, skip_special_tokens=True).split()[-1] for output in outputs]

        # Ensure the generated word is not the same as the original word
        for generated_word in generated_words:
            if generated_word != original_words[mask_indices.index(idx)]:
                replacement_words.append(generated_word)
                break
        else:
            # If all generated words are the same as the original, pick the first one as a fallback
            replacement_words.append(generated_words[0])

    # Replace the masked words with the generated words
    for idx, replacement in zip(mask_indices, replacement_words):
        masked_words[idx] = replacement

    final_augmented_text = ' '.join(masked_words)

    # Print statements to show the augmentation process
    print(f"Original sentence: {' '.join(words)}")
    print(f"Masked words: {original_words}")
    print(f"Generated replacement words: {replacement_words}")
    print(f"New augmented sentence: {final_augmented_text}")

    return final_augmented_text

# Function to prepare and save the augmented data
def prepare_and_save_augmented_data(augmentation_method, postag=None):
    model_path = './models/fine_tuned_GPT2_2016_model'
    tokenizer_path = './models/fine_tuned_GPT2_2016_tokenizer'

    # Load the fine-tuned model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()

    augmented_sentences = []

    for text_with_placeholder, target, polarity, from_index, to_index in sentences:
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

          # Prepend "positive" or "negative" based on polarity
          if polarity == '1':
             words = ['positive'] + words
          elif polarity == '0':
             words = ['neutral'] + words
          elif polarity == '-1':
             words = ['negative'] + words

          # Update mask indices to account for the prepended word
          mask_indices = [idx + 1 for idx in mask_indices]

          # Fill masked words
          augmented_text = fill_masked_words(mask_indices, words, model, tokenizer)

          # Remove the prepended word
          augmented_words = augmented_text.split()[1:]

          # Find and replace the target with $T$
          augmented_with_placeholder = ' '.join(augmented_words).replace(target, '$T$', 1)

          augmented_sentences.append((augmented_with_placeholder, target, polarity))


    if augmentation_method == 'POS':
        augmented_txt_path = os.path.join(augmented_data_path, f'GPT2_2016augmented_data_{augmentation_method}_{postag}.txt')
    else:
        augmented_txt_path = os.path.join(augmented_data_path, f'GPT2_2016augmented_data_{augmentation_method}.txt')

    with open(augmented_txt_path, 'w') as f:
        for text, target, polarity in augmented_sentences:
            f.write(f"{text}\n{target}\n{polarity}\n")

    print(f"Augmented dataset successfully saved to {augmented_txt_path}")

# Create augmented data files for different methods
prepare_and_save_augmented_data('Random')
prepare_and_save_augmented_data('Sentiment')
for pos_tag in ['VB', 'JJ', 'NN', 'RB', 'Combi']:
    prepare_and_save_augmented_data('POS', postag=pos_tag)
