import osmnx as ox
import networkx as nx
import torch
from torch_geometric.data import Data

# === Utility to safely convert to float ===
def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0

# === Load labeled graph ===
G = ox.load_graphml("data/dhaka_dhanmondi_labeled.graphml")
G = G.to_undirected()

# Relabel nodes to integer indices for PyTorch Geometric
G = nx.convert_node_labels_to_integers(G, label_attribute="osmid")
node_mapping = {n: i for i, n in enumerate(G.nodes())}

# === Prepare edge_index, features, and labels ===
edge_index = []
edge_attr = []
edge_labels = []

for u, v, data in G.edges(data=True):
    edge_index.append([node_mapping[u], node_mapping[v]])

    # Features: [length, betweenness] with safe casting
    length = safe_float(data.get("length", 0))
    betweenness = safe_float(data.get("betweenness", 0))
    edge_attr.append([length, betweenness])

    # Label
    label = int(data.get("importance_label", 0))
    edge_labels.append(label)

# === Convert to tensors ===
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attr, dtype=torch.float)
edge_labels = torch.tensor(edge_labels, dtype=torch.long)

# Dummy node features (e.g., ones)
x = torch.ones((G.number_of_nodes(), 1), dtype=torch.float)

# === Create PyG Data object ===
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=edge_labels)

# === Save to disk ===
torch.save(data, "data/pyg_dhanmondi.pt")
print("Saved PyTorch Geometric graph to 'data/pyg_dhanmondi.pt'")

