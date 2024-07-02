import torch
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling

# Define the path to the text file
text_file = '/content/drive/MyDrive/Thesis/DataAugmentation/data/processedData/data_16_train.txt'

# Function to read the text file and extract data
def parse_text(file):
    texts = []
    with open(file, 'r') as f:
        lines = f.readlines()
        previous_text = None
        for i in range(0, len(lines), 3):
            text = lines[i].replace('$T$', lines[i+1].strip()).strip()
            if text != previous_text: # Consider sentences only ones when they are new and not multiple times in the file (when they have multiple targets)
                texts.append(text)
                previous_text = text
    return texts

# Extract text data from the text file
texts = parse_text(text_file)

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
    eval_strategy="epoch",  
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
model.save_pretrained('./models/fine_tuned_BERT_2016_model')
tokenizer.save_pretrained('./models/fine_tuned_BERT_2016_tokenizer')
