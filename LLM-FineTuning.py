import pandas as pd
from datasets import Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments


data = {
    "text": ["నేను మీకు ఎలా సహాయం చేయగలను", "నాకు ఈరోజు వార్తలు చెప్పు", "సుప్రభాతం"],
    "intent": ["How can I help you?", "Todays news", "greet"]
}

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)


model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)


train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']


num_labels = len(data['intent'])
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


trainer.train()


model.save_pretrained('./fine-tuned-alexa')
tokenizer.save_pretrained('./fine-tuned-alexa')
