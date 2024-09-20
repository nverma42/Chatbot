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

### Goals

1. **Implement validation loop & Add function to evaluate model on validation set**:
A validation loop is crucial for assessing how well your model generalizes to unseen data. It helps prevent overfitting.

```python
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            src = batch['input_ids']
            tgt = batch['labels']
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, model.fc_out.out_features), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)
```

You would call this function after each epoch in your training loop.

2. **Add early stopping mechanism**:
Early stopping helps prevent overfitting by stopping the training process when the model's performance on the validation set stops improving.

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
```

3. **Implement learning rate scheduling:**
Learning rate scheduling can help improve model convergence. PyTorch provides several schedulers:

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Reduce LR on plateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100)
```

4. **Add support for gradient clipping:**
Gradient clipping helps prevent exploding gradients, which can cause training instability:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This would be called after `loss.backward()` but before `optimizer.step()`.

5. **Implement logging and metrics tracking:**
TensorBoard is a great tool for visualizing training progress. Here's how you might use it:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

# In your training loop:
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/validation', val_loss, epoch)
writer.add_scalar('Learning rate', scheduler.get_last_lr()[0], epoch)
```

6. **Add support for distributed training**:
Distributed training allows you to use multiple GPUs efficiently. PyTorch's `DistributedDataParallel` is commonly used for this:

```python
model = torch.nn.parallel.DistributedDataParallel(model)
```

7. **Implement custom loss functions:**
Custom loss functions can be useful for specific tasks. Here's an example of label smoothing:

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
```

8. **Add type hints:**
Type hints improve code readability and can catch certain types of errors early.

```python
def train(model: nn.Module, 
          data_loader: DataLoader, 
          epochs: int, 
          v_size: int, 
          lr: float) -> None:
    # ... implementation ...
```
