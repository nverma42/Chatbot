import torch
import torch.nn as nn
import pandas as pd
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# The dataset must have context and response columns in CSV format.


class TextDataSet(Dataset):
    """
    Custom Dataset class for handling text data with context and response columns.

    Attributes:
        file_path (str): The file path to the CSV dataset.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to preprocess text data.
        max_length (int): Maximum sequence length for tokenization. Default is 128.
    """

    def __init__(self, file_path, tokenizer, max_length=128):
        """
        Initializes the dataset by reading the CSV file and setting tokenizer and max_length.

        Args:
            file_path (str): Path to the dataset CSV file.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer object used to process text.
            max_length (int): Maximum sequence length for tokenization. Default is 128.
        """
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = 128

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of rows in the CSV file.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a single sample from the dataset, which consists of a context and response pair.
        The text is tokenized using the provided tokenizer and max_length.

        Args:
            index (int): Index of the desired sample.

        Returns:
            dict: Dictionary containing tokenized context and response.
        """
        context = self.data.iloc[index][0]
        response = self.data.iloc[index][1]

        # Tokenize
        context_encoding = self.tokenizer(context,
                                          truncation=True,  # Truncate sequence if it is bigger than max length
                                          padding='max_length',  # Ensure text sequence is padded to max length
                                          max_length=self.max_length,  # Text max length in number of words
                                          return_tensors='pt')  # Set the format of the return tensors is pytorch

        response_encoding = self.tokenizer(response,
                                           truncation=True,  # Truncate sequence if it is bigger than max length
                                           padding='max_length',  # Ensure text sequence is padded to max length
                                           max_length=self.max_length,  # Text max length in number of words
                                           return_tensors='pt')  # Set the format of the return tensors is pytorch

        input_ids = context_encoding['input_ids'].squeeze()
        attention_mask = context_encoding['attention_mask'].squeeze()
        labels = response_encoding['input_ids'].squeeze()

        return {'input_ids': input_ids, 'labels': labels}


class TransformerModel(nn.Module):
    # v_size - Vocabulary size needed for defining embedding for words
    # d_model - Size of the word vector
    # n_heads - The number of attention heads
    # num_layers - The number of layers in encoders and decoders
    # max_seq_length - Maximum sequence length
    def __init__(self, v_size, d_model, n_heads, num_layers, max_seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(v_size, d_model)

        # Learnable positional encoding. Weights are initialized to zeros.
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_length, d_model))

        # Define the transformer model
        self.transformer = nn.Transformer(
            d_model, n_heads, num_layers, num_layers)

        # Project each word of size d_model to the vocabulary space
        self.fc_out = nn.Linear(d_model, v_size)

    def forward(self, source, target):
        # Combine the embedding as well as positional encoding to create more meaningful embedding
        src = self.embedding(source) + \
            self.positional_encoding[:, source.size(1)]
        tgt = self.embedding(target) + \
            self.positional_encoding[:, target.size(1)]

        output = self.transformer(src, tgt)

        # Apply linear layer independently to each token in the output and project it to the vocabulary space
        return (self.fc_out(output))


def train(model, data_loader, epochs, v_size, lr):
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        try:
            for batch in data_loader:
                # Reset all the gradients for the current iteration
                optimizer.zero_grad()

                src = batch['input_ids']
                tgt = batch['labels']

                output = model(src, tgt)
                loss = criterion(output.view(-1, v_size), tgt.view(-1))

                loss.backward()

                # Update the weights
                optimizer.step()
        except Exception as e:
            print(e)


def generate_text(model, tokenizer, start_text, max_seq_length):
    model.eval()
    input_ids = torch.tensor(tokenizer.tokenize(start_text)).unsqueeze(0)
    output_ids = input_ids

    output_text = ""
    with torch.no_grad():
        for _ in max_seq_length:
            output = model(input_ids, output_ids)
            predicted_id = output[0, -1].argmax().item()
            # Convert the predicted index to a word in the vocabulary
            predicted_word = tokenizer.decode([predicted_id])
            output_text = output_text + " " + predicted_word

    return output_text


# Main Flow
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

train_ds = TextDataSet("./test.csv", tokenizer)

# Define data loaders to visualize the data
train_dl = DataLoader(dataset=train_ds, batch_size=4,
                      shuffle=True, drop_last=True)

# Initialize the model
model = TransformerModel(v_size=tokenizer.vocab_size,
                         d_model=512, n_heads=8, num_layers=6, max_seq_length=128)

# Train the model
train(model, train_dl, 5, tokenizer.vocab_size, 0.01)

# Generate text
