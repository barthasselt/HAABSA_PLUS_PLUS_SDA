import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq

# Define the path to the text file
text_file = '/content/drive/MyDrive/Thesis/DataAugmentation/data/processedData/data_15_train.txt'

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

# Load the pre-trained BART model and tokenizer
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Tokenize the datasets
def tokenize_function(examples):
    inputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
    outputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
    inputs['labels'] = outputs['input_ids']
    return inputs

# Tokenize the dataset before splitting
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Split the tokenized dataset into 80% train and 20% test
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

# Define the data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
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
model.save_pretrained('./models/fine_tuned_BART_2015_model')
tokenizer.save_pretrained('./models/fine_tuned_BART_2015_tokenizer')
