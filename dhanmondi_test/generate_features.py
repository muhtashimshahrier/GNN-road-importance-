import osmnx as ox
import networkx as nx

# Load the saved graph
G = ox.load_graphml("data/dhaka_dhanmondi.graphml")

# Convert to undirected for centrality calculation
G_undirected = G.to_undirected()

# Compute node-level betweenness centrality
node_betweenness = nx.betweenness_centrality(G_undirected, weight='length', normalized=True)

# Assign edge-level betweenness: average of node centralities
for u, v, k, data in G.edges(keys=True, data=True):
    b_u = node_betweenness.get(u, 0)
    b_v = node_betweenness.get(v, 0)
    data["betweenness"] = (b_u + b_v) / 2

ox.save_graphml(G, "data/dhaka_dhanmondi_features.graphml")

print("Edge features added and graph saved.")
