import torch
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
import xml.etree.ElementTree as ET

# Define the path to the txt file
text_file = '/content/drive/MyDrive/Thesis/DataAugmentation/data/processedData/data_16_train.txt'

# Function to prepend sentiment and extract texts
def prepend_sentiment_and_extract(input_file):
    sentiment_map = {-1: 'negative', 0: 'neutral', 1: 'positive'}
    texts = []
    previous_text = None

    with open(input_file, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 3):
        sentence = lines[i].strip()
        sentiment_value = int(lines[i + 2].strip())
        sentiment = sentiment_map.get(sentiment_value, 'unknown')
        prepended_sentence = f"{sentiment} {sentence}\n"
        
        # Replace $T$ with the target text
        text = prepended_sentence.replace('$T$', lines[i + 1].strip()).strip()
        if text != previous_text:
            texts.append(text)
            previous_text = text

    return texts

# Call the function and get texts
texts = prepend_sentiment_and_extract(text_file)

# Create a dataset from the extracted texts
data = {"text": texts}
dataset = Dataset.from_dict(data)


# Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# Tokenize the dataset before splitting
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Split the tokenized dataset into 80% train and 20% test
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed = 530)
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

# Define the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./models",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    eval_strategy="epoch",  # Use eval_strategy instead of deprecated evaluation_strategy
)

# Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./models/fine_tuned_PREPENDBERT_2016_model')
tokenizer.save_pretrained('./models/fine_tuned_PREPENDBERT_2016_tokenizer')
