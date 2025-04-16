import osmnx as ox
import networkx as nx
from tqdm import tqdm

G = ox.load_graphml("data/dhaka_labeled.graphml")

G_undirected = G.to_undirected()

node_betweenness = nx.betweenness_centrality(
    G_undirected, weight="length", normalized=True, k=1000, seed=42
)

for u, v, k, data in tqdm(G.edges(keys=True, data=True), total=G.number_of_edges()):
    b_u = node_betweenness.get(u, 0.0)
    b_v = node_betweenness.get(v, 0.0)
    data["betweenness"] = (b_u + b_v) / 2


ox.save_graphml(G, "data/dhaka_features.graphml")
print("Features added and saved as 'data/dhaka_features.graphml'")
