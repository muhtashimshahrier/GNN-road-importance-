import osmnx as ox

# Load the feature-rich graph
G = ox.load_graphml("data/dhaka_dhanmondi_features.graphml")

# Define binary label map
label_map = {
    "residential": 0,
    "unclassified": 0,
    "secondary": 1,
    "secondary_link": 1
}

# Add label field to each edge
for u, v, k, data in G.edges(keys=True, data=True):
    tag = data.get("highway")
    
    # Handle list tags
    if isinstance(tag, list):
        tag = tag[0]

    # Assign label
    label = label_map.get(tag, 0)  # default to 0 if unknown
    data["importance_label"] = label

# Save labeled graph
ox.save_graphml(G, "data/dhaka_dhanmondi_labeled.graphml")

print("Labeled graph saved with 'importance_label' field on edges.")
