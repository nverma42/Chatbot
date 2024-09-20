# ChatBot Training Dataset Handler

This repository provides a simple framework for processing text datasets used in chatbot models. The code is designed to handle a dataset consisting of context-response pairs stored in CSV format. The `TextDataSet` class tokenizes the text data using the HuggingFace `GPT2Tokenizer` and prepares it for use in PyTorch's `DataLoader`.

## Features

- Custom Dataset class (`TextDataSet`) for handling text data.
- Tokenization support using `transformers` library.
- Prepares the dataset for training chatbot models in PyTorch.

## Requirements

- Python 3.8+
- PyTorch
- transformers (HuggingFace)
- pandas

## Setup

### Using Conda

```bash
# Create a conda environment
conda create --name chatbot-env python=3.8

# Activate the environment
conda activate chatbot-env

# Install dependencies
conda install pytorch torchvision torchaudio -c pytorch
pip install transformers pandas
```

### using venv

```bash
# Create a virtual environment
python3 -m venv chatbot-env

# Activate the environment
source chatbot-env/bin/activate  # On Windows use chatbot-env\Scripts\activate

# Install dependencies
pip install torch transformers pandas
```

### Usage

- Prepare a CSV file with two columns: context and response.
- Use the TextDataSet class to load and process the data.
