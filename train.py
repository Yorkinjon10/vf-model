# train.py
import torch
import torch.nn as nn
import torch.optim as optim

# 1️⃣ Load text
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2️⃣ Create vocabulary
chars = sorted(list(set(text)))  # unique characters
vocab_size = len(chars)

# 3️⃣ Mapping characters to numbers
char_to_idx = { ch:i for i,ch in enumerate(chars) }
idx_to_char = { i:ch for i,ch in enumerate(chars) }

print(f"Vocab size: {vocab_size}")

# Encode entire text as numbers
data = [char_to_idx[ch] for ch in text]

# Create input-target pairs
X = data[:-1]  # all chars except last
Y = data[1:]   # next char as target

X = torch.tensor(X)
Y = torch.tensor(Y)

print("Example input:", X[:10])
print("Example target:", Y[:10])

class TinyCharModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Embedding layer: converts number → vector
        self.embedding = nn.Embedding(vocab_size, 16)  # 16-dim embedding
        # Linear layer: predict next char
        self.fc = nn.Linear(16, vocab_size)

    def forward(self, x):
        x = self.embedding(x)   # x: [batch] → [batch, 16]
        x = self.fc(x)          # x: [batch, vocab_size]
        return x

# Model
model = TinyCharModel(vocab_size)
# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
X = X.to(device)
Y = Y.to(device)


epochs = 200  # small dataset → more epochs

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


# Start with a random character
start_char = "h"
input_idx = torch.tensor([char_to_idx[start_char]]).to(device)

generated = start_char

for _ in range(100):
    with torch.no_grad():
        output = model(input_idx)
        probs = torch.softmax(output, dim=1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        next_char = idx_to_char[next_idx]
        generated += next_char
        input_idx = torch.tensor([next_idx]).to(device)

print("Generated text:\n", generated)

