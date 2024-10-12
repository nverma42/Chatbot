from multiprocessing import context
import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import re

# To Do:
# 1. Make a simple model work.
# 2. Think about how would we generate vocabulary.
# 3. Add positional encoding.

class TransformerModel(nn.Module):
    def __init__(self, max_seq_len, input_dim,  output_dim, hidden_dim, n_layers, n_heads, dropout):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Embedding(input_dim, hidden_dim)
        self.input_positional_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        self.decoder = nn.Embedding(output_dim, hidden_dim)
        self.output_positional_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        self.transformer = nn.Transformer(hidden_dim, n_heads, n_layers, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg):
        input_seq_len = src.size(1)
        src = self.encoder(src) + self.input_positional_embeddings[:, :input_seq_len, :]     
        output_seq_len = trg.size(1)
        trg = self.decoder(trg) + self.output_positional_embeddings[:, :output_seq_len, :]
        output = self.transformer(src, trg)
        output = self.fc_out(output)
        return output


# Text normalization
def tokenize_text(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize
    tokens = text.split(' ')
    return tokens

# Main driver

# Generate vocabulary
df = pd.read_csv('./test - Copy.csv')
contexts = []
responses = []
for index, row in df.iterrows():
    contexts.append(row['Context'])
    responses.append(row['Response'])

# Normalize text and add to the vocabulary
vocab = dict()
word_decoder = dict()
input_tokens = []
output_tokens = []
code = 1
for i in range(len(df)):
    tokens = tokenize_text(contexts[i])
    input_tokens.append(tokens)
    for token in tokens:
        if (token not in vocab):
            vocab[token] = code
            word_decoder[code] = token
            code += 1

    tokens = tokenize_text(responses[i])
    output_tokens.append(tokens)
    for token in tokens:
        if (token not in vocab):
            vocab[token] = code
            word_decoder[code] = token
            code += 1
        
# Add the out of vocabulary word
vocab['OOV'] = code
word_decoder[code] = 'OOV'

# Set various parameters
max_seq_len = 16
input_dim = len(vocab)   # Vocabulary size for input
output_dim = len(vocab)  # Vocabulary size for output
hidden_dim = 512
n_layers = 6
n_heads = 8
dropout = 0.1

# Tokenize and convert to tensor
encoded_input_tokens = []   # Example tokenized input
for tokens in input_tokens:
    encoded_tokens = [vocab[token] for token in tokens]
    encoded_tokens = encoded_tokens + (max_seq_len - len(encoded_tokens)) * [0]
    encoded_input_tokens.append(encoded_tokens)

encoded_output_tokens = []  # Example tokenized output
for tokens in output_tokens:
    encoded_tokens = [vocab[token] for token in tokens]
    encoded_tokens = encoded_tokens + (max_seq_len - len(encoded_tokens)) * [0]
    encoded_output_tokens.append(encoded_tokens)
    
input_tensor = torch.tensor(encoded_input_tokens)
output_tensor =  torch.tensor(encoded_output_tokens)

model = TransformerModel(max_seq_len, input_dim, output_dim, hidden_dim, n_layers, n_heads, dropout)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(input_tensor, output_tensor)
    loss = criterion(output.view(-1, output_dim), output_tensor.view(-1))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate
model.eval()
input_tensor = torch.tensor([encoded_input_tokens[2]])
print('Context: ' + str(input_tokens[2]))
generated_tokens = input_tensor.clone()

output = []
with torch.no_grad():
    logits = model(input_tensor, generated_tokens)
    predicted_tokens = torch.argmax(logits, dim=-1)
    output = predicted_tokens.numpy()[0]

decoded_text = [word_decoder[code] for code in output]
print('Response: ' + str(decoded_text))