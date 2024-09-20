import pandas as pd
from torch.utils.data import Dataset


class TextDataSet(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
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
