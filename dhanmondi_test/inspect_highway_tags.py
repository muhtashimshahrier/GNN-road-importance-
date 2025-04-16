import osmnx as ox
from collections import Counter

# Load the graph with features
G = ox.load_graphml("data/dhaka_dhanmondi_features.graphml")

# Count all highway tag types
highway_tags = []

for u, v, k, data in G.edges(keys=True, data=True):
    hwy = data.get("highway")
    if isinstance(hwy, list):
        highway_tags.extend(hwy)
    elif isinstance(hwy, str):
        highway_tags.append(hwy)

# Display counts
tag_counts = Counter(highway_tags)

print("Unique highway tag types and counts:")
for tag, count in tag_counts.items():
    print(f"{tag}: {count}")
