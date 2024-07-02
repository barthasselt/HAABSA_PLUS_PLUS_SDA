import torch
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
import nltk
nltk.download('punkt')

# Define the path to the text file for fine-tuning
text_file = '/content/drive/MyDrive/Thesis/DataAugmentation/data/processedData/data_16_train.txt'

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
model = CustomBertForMaskedLM.from_pretrained('bert-base-uncased', ignore_mismatched_sizes=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

# Load and preprocess the data
sentences = parse_txt(text_file)
if not sentences:
    print("No sentences found in the text file.")
    exit()

class CustomDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_len):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence, target, polarity = self.sentences[idx]
        inputs = self.tokenizer(sentence, return_tensors='pt', max_length=self.max_len, padding='max_length', truncation=True)
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        
        # Encode polarity
        polarity_mapping = {-1: 0, 0: 1, 1: 2}
        token_type_ids = torch.tensor([polarity_mapping[int(polarity)]] * self.max_len)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': input_ids
        }

max_len = 128
dataset = CustomDataset(sentences, tokenizer, max_len)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=530)

# Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Set up training configuration
training_args = TrainingArguments(
    output_dir='./models',
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch",
    logging_dir='/content/drive/MyDrive/Thesis/DataAugmentation/logs',
    logging_steps=200,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model('/content/drive/MyDrive/Thesis/DataAugmentation/models/fine_tuned_CBERT2016_model')
tokenizer.save_pretrained('/content/drive/MyDrive/Thesis/DataAugmentation/models/fine_tuned_CBERT2016_tokenizer')

print("Fine-tuned model and tokenizer saved successfully.")
