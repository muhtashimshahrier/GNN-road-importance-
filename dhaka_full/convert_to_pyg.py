import osmnx as ox
import networkx as nx
import torch
from torch_geometric.data import Data

G = ox.load_graphml("data/dhaka_labeled.graphml")
G = nx.Graph(G)  # Remove direction + multiedges

G = nx.convert_node_labels_to_integers(G, label_attribute="osmid")
node_mapping = {n: i for i, n in enumerate(G.nodes())}

# Build edge-level tensors
edge_index_list = []
edge_attr_list = []
edge_labels_list = []

for u, v, data in G.edges(data=True):
    u_id = node_mapping[u]
    v_id = node_mapping[v]

    edge_index_list.append([u_id, v_id])

    length = float(data.get("length", 0))
    betweenness = float(data.get("betweenness", 0))
    edge_attr_list.append([length, betweenness])

    label = int(data.get("importance_label", 0))
    edge_labels_list.append(label)

# Convert to PyTorch tensors
edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
edge_labels = torch.tensor(edge_labels_list, dtype=torch.long)
x = torch.ones((G.number_of_nodes(), 1), dtype=torch.float)  # dummy node features

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    edge_labels=edge_labels,
    num_nodes=G.number_of_nodes()
)

torch.save(data, "data/pyg_dhaka.pt")
print("Saved 'pyg_dhaka.pt' with matching shapes.")
