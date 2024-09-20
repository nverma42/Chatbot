# ChatBot Training Dataset Handler

This repository contains code to train and generate text using a custom Transformer model. It includes a custom dataset loader, model definition, and training utilities for text generation.

## Features

- Transformer-based model for text generation.
- Custom dataset handling for context-response pairs.
- Training utilities for handling model training and text generation.

## Requirements

- Python 3.8+
- PyTorch
- transformers (HuggingFace)
- pandas
- absl-py

## Setup

### Using Conda

```bash
# Create a conda environment
conda create --name chatbot-env python=3.8

# Activate the environment
conda activate chatbot-env

# Install dependencies
conda install pytorch torchvision torchaudio -c pytorch
pip install transformers pandas absl-py
```

### using venv

```bash
# Create a virtual environment
python3 -m venv chatbot-env

# Activate the environment
source chatbot-env/bin/activate  # On Windows use chatbot-env\Scripts\activate

# Install dependencies
pip install torch transformers pandas absl-py
```

### Usage

- Prepare a CSV file with two columns: context and response.
- Use the TextDataSet class to load and process the data.

```python
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from transformer_model import TransformerModel
from train_utils import train, generate_text

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Create the dataset and dataloader
train_ds = TextDataSet('path_to_your_dataset.csv', tokenizer)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

# Initialize and train the model
model = TransformerModel(v_size=len(tokenizer), d_model=512, n_heads=8, num_layers=6, max_seq_length=128)
train(model, train_dl, epochs=5, v_size=len(tokenizer), lr=0.001)

# Generate text from the trained model
generated_text = generate_text(model, tokenizer, start_text="Once upon a time", max_seq_length=50)
print(generated_text)
```