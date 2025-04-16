import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter

data = torch.load("data/pyg_dhaka.pt")

X = data.edge_attr            # [num_edges, 2] (length + betweenness)
y = data.y                    # [num_edges]

print("Class distribution:", Counter(y.tolist()))

indices = list(range(X.shape[0]))
train_idx, test_idx = train_test_split(indices, test_size=0.3, stratify=y, random_state=42)

# Define model
model = Sequential(
    Linear(X.shape[1], 32),
    ReLU(),
    Linear(32, 16),
    ReLU(),
    Linear(16, 4)  # 4 classes
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(X)
    loss = loss_fn(out[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

model.eval()
pred = model(X)
pred_labels = pred.argmax(dim=1)

print("\n=== Classification Report (Test Set) ===")
print(classification_report(y[test_idx].cpu(), pred_labels[test_idx].cpu()))
