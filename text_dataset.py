import pandas as pd
from torch.utils.data import Dataset

# The dataset must have context and response columns in CSV format.


class TextDataSet(Dataset):
    """
    A PyTorch Dataset for loading and tokenizing text data with context-response pairs.

    Attributes:
        file_path (str): Path to the CSV file containing context-response pairs.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for text data.
        max_length (int): Maximum sequence length for tokenization.
    """

    def __init__(self, file_path, tokenizer, max_length=128):
        """
        Initializes the dataset by reading the CSV file and setting tokenizer and max_length.

        Args:
            file_path (str): Path to the dataset CSV file.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer object for tokenization.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of rows in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset and tokenizes it.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing tokenized context and response.
        """
        context = self.data.iloc[index]['context']
        response = self.data.iloc[index]['response']

        context_encoding = self.tokenizer(context,
                                          truncation=True,
                                          padding='max_length',
                                          max_length=self.max_length,
                                          return_tensors='pt')

        response_encoding = self.tokenizer(response,
                                           truncation=True,
                                           padding='max_length',
                                           max_length=self.max_length,
                                           return_tensors='pt')

        input_ids = context_encoding['input_ids'].squeeze()
        attention_mask = context_encoding['attention_mask'].squeeze()
        labels = response_encoding['input_ids'].squeeze()

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
