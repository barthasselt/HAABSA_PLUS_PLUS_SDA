import xml.etree.ElementTree as ET
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset
from transformers import Trainer, TrainingArguments

# Define the path to the txt file
text_file = '/content/drive/MyDrive/Thesis/DataAugmentation/data/processedData/data_16_train.txt'

# Function to prepend sentiment, extract texts, and format them
def prepend_sentiment_and_extract(input_file, sep_token='', eos_token=''):
    sentiment_map = {-1: 'negative', 0: 'neutral', 1: 'positive'}
    texts = []
    previous_text = None

    with open(input_file, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 3):
        sentence = lines[i].strip()
        sentiment_value = int(lines[i + 2].strip())
        sentiment = sentiment_map.get(sentiment_value, 'unknown')
        
        # Replace $T$ with the target text
        text = sentence.replace('$T$', lines[i + 1].strip()).strip()
        
        # Formatted text: polarity - sep_token - text - eos_token
        formatted_text = f"{sentiment} {sep_token} {text} {eos_token}"
        
        if formatted_text != previous_text:
            texts.append(formatted_text)
            previous_text = formatted_text

    return texts

# Call the function and get texts
finetuning_data = prepend_sentiment_and_extract(text_file, sep_token=' <SEP> ', eos_token=' <EOS>')

# Create dataset dictionary
dataset_dict = {"text": finetuning_data}

# Create Hugging Face Dataset
dataset = Dataset.from_dict(dataset_dict)

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add a padding token
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = inputs["input_ids"].copy()  # Copy input_ids to labels
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split the tokenized dataset into 80% train and 20% test
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

# Prepare the datasets for training
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# Load the pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Resize the model's embedding to match the new tokenizer size
model.resize_token_embeddings(len(tokenizer))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./models",  # output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=10,  # number of training epochs
    per_device_train_batch_size=8,  # batch size for training
    per_device_eval_batch_size=8,  # batch size for evaluation
    save_steps=10_000,  # after how many steps to save a model checkpoint
    save_total_limit=2,  # limit the total amount of checkpoints
    evaluation_strategy="epoch",  # evaluation strategy
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./models/fine_tuned_GPT2_2016_model')
tokenizer.save_pretrained('./models/fine_tuned_GPT2_2016_tokenizer')
