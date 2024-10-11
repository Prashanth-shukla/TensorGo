import pandas as pd
from datasets import Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments

# Load your Telugu dataset
data = {
    "text": ["వాతావరణం ఎలా ఉంది?", "నాకు మ్యూజిక్ వాడు", "సుప్రభాతం"],
    "intent": ["ask_weather", "play_music", "greet"]
}

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# Load the tokenizer
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split dataset into training and testing sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Load the pre-trained model
num_labels = len(data['intent'])
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine-tuned-alexa-telugu')
tokenizer.save_pretrained('./fine-tuned-alexa-telugu')
