import osmnx as ox

place_name = "Dhaka, Bangladesh"

# Download and save the full drivable road network
G = ox.graph_from_place(place_name, network_type="drive")

ox.save_graphml(G, "data/dhaka.graphml")
print(f"Saved full Dhaka graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")
