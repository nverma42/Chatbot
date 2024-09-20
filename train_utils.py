import torch
import torch.nn as nn
import torch.optim as optim


def train(model, data_loader, epochs, v_size, lr):
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch in data_loader:
            optimizer.zero_grad()
            src = batch['input_ids']
            tgt = batch['labels']
            output = model(src, tgt[:, :-1])  # Shift target sequence
            loss = criterion(output.reshape(-1, v_size),
                             tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


def generate_text(model, tokenizer, start_text, max_seq_length):
    model.eval()
    input_ids = tokenizer.encode(start_text, return_tensors='pt')
    output_ids = input_ids

    with torch.no_grad():
        for _ in range(max_seq_length):
            output = model(input_ids, output_ids)
            predicted_id = output[0, -1].argmax().item()
            output_ids = torch.cat(
                [output_ids, torch.tensor([[predicted_id]])], dim=-1)
            if predicted_id == tokenizer.eos_token_id:
                break

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
