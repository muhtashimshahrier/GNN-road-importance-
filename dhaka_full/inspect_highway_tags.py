import osmnx as ox
from collections import Counter

G = ox.load_graphml("data/dhaka.graphml")

# Count all highway tags
highway_tags = []

for u, v, k, data in G.edges(keys=True, data=True):
    hwy = data.get("highway")
    if isinstance(hwy, list):
        highway_tags.extend(hwy)
    elif isinstance(hwy, str):
        highway_tags.append(hwy)

tag_counts = Counter(highway_tags)

print("Highway tag types and their counts:")
for tag, count in tag_counts.items():
    print(f"{tag}: {count}")
