import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import numpy as np


data = torch.load("data/pyg_dhaka.pt")
edge_index = data.edge_index
edge_attr = data.edge_attr
edge_labels = data.edge_labels
x = data.x

# Build graph structure
src, dst = edge_index

indices = list(range(edge_labels.shape[0]))
train_idx, test_idx = train_test_split(
    indices, test_size=0.3, stratify=edge_labels.numpy(), random_state=42
)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(edge_labels.numpy()),
    y=edge_labels.numpy()
)
class_weights = torch.tensor(class_weights, dtype=torch.float)

print("Class distribution:", Counter(edge_labels.tolist()))
print("Class weights:", class_weights.tolist())

# GCN-based edge classifier
class EdgeClassifier(nn.Module):
    def __init__(self, in_channels_node, in_channels_edge, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels_node, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2 + in_channels_edge, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst], edge_attr], dim=1)
        out = self.classifier(edge_features)
        return out

# Initialize model
model = EdgeClassifier(
    in_channels_node=x.shape[1],
    in_channels_edge=edge_attr.shape[1],
    hidden_channels=32,
    num_classes=4
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index, edge_attr)
    loss = loss_fn(out[train_idx], edge_labels[train_idx])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    pred = model(x, edge_index, edge_attr).argmax(dim=1)

print("\n=== Classification Report (Test Set) ===")
print(classification_report(edge_labels[test_idx].cpu(), pred[test_idx].cpu()))
