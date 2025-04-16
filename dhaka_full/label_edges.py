import osmnx as ox

G = ox.load_graphml("data/dhaka.graphml")

label_map = {
    "motorway": 3,
    "trunk": 3,
    "motorway_link": 3,
    "trunk_link": 3,
    
    "primary": 2,
    "primary_link": 2,

    "secondary": 1,
    "secondary_link": 1,
    "tertiary": 1,
    "tertiary_link": 1,

    "residential": 0,
    "living_street": 0,
    "unclassified": 0,
    "road": 0
}

for u, v, k, data in G.edges(keys=True, data=True):
    tag = data.get("highway")
    
    
    if isinstance(tag, list):
        tag = tag[0]

    label = label_map.get(tag, 0)  # default to 0 if unknown
    data["importance_label"] = label

ox.save_graphml(G, "data/dhaka_labeled.graphml")
print("Labeled Dhaka graph saved as 'data/dhaka_labeled.graphml'")
