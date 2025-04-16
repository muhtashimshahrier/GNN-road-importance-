import osmnx as ox

place_name = "Dhanmondi Residential Area, Dhaka, Bangladesh"

G = ox.graph_from_place(place_name, network_type="drive")


ox.save_graphml(G, "data/dhaka_dhanmondi.graphml")

print(f"Graph saved with {len(G.nodes)} nodes and {len(G.edges)} edges.")
