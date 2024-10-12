from unittest.util import _MAX_LENGTH
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import EncoderDecoderModel, BertTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Sample data: context-response pairs
data = [
    ("What resources are available for mental health", "Depression"),
    ("What parts of brain are affected in depression and anxiety", "Biology"),
    ("Where do you live?", "Query"),
    ("I am feeling very low today", "Sad"),
    ("I got fired from my job today.",	"Anxious"),
    ("I feel bad when my kids let me down",	"Furious"),
    ("I had fight in the family today",	"Angry"),
    ("My family does not talk to me",	"Frustrated")

]

# Custom dataset class
class ChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, response = self.data[idx]
        inputs = self.tokenizer.encode_plus(
            context,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        outputs = self.tokenizer.encode_plus(
            response,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        labels = outputs['input_ids'].squeeze()
        return input_ids, attention_mask, labels

# Initialize tokenizer and model
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')

# checkpoint 
checkpoint = "gpt2"
# download and cache tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

# download and cache pre-trained model
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Create dataset and dataloader
dataset = ChatDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training parameters
epochs = 5
learning_rate = 1e-2

# Optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(epochs):
    for input_ids, attention_mask, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        #labels = torch.zeros(logits.size(0), dtype=torch.long)  # Dummy labels for demonstration
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        #loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs} completed. Loss: {loss.item()}")

print("Training completed.")

# Example inference
model.eval()
context = "I feel very low today"
inputs = tokenizer.encode_plus(context, add_special_tokens=True, return_tensors="pt")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
outputs = outputs.squeeze().tolist()
response = tokenizer.decode(outputs, skip_special_tokens=True)
print(f"Context: {context}")
print(f"Response: {response}")




