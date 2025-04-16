# Road Segment Importance Prediction using Graph Neural Networks

This project predicts the relative **importance of road segments** in the urban road network of **Dhaka, Bangladesh** using a Graph Neural Network (GNN) pipeline.

We extract a real road network using **OSMnx**, engineer graph-based features, assign multi-class importance labels using OpenStreetMap `highway` tags, and train a GNN (MLP baseline) using **PyTorch Geometric**.

---

## Folder Structure

GNN-road-importance/
├── dhanmondi_test/      # Small-scale prototype
├── dhaka_full/          # Full Dhaka road network (~250k edges)
└── README.md            # Project overview and documentation


---

## Tools Used

- Python
- PyTorch Geometric
- OSMnx
- NetworkX
- scikit-learn
- tqdm

---

## What This Project Does

| Step               | Description |
|--------------------|-------------|
| Graph Download   | Retrieve road network using OSMnx |
| Labeling         | Assign 4-class importance from `highway` tag |
| Feature Engineering | Compute edge `length` and `betweenness centrality` |
| PyG Conversion   | Convert NetworkX graph to PyTorch Geometric |
| Model Training   | Train MLP model to classify road importance |
| Class Balancing  | Use weighted loss to improve rare class prediction |

---

## Folder Overview

- `dhanmondi_test/`: Prototype using a small subgraph (~2k edges)
- `dhaka_full/`: Full-scale Dhaka road network (~250k edges, 100k nodes)

Each folder includes:
- `download_network.py`
- `label_edges.py`
- `generate_features.py`
- `convert_to_pyg.py`
- `train_gnn_baseline.py`
- `train_gnn_weighted.py` (Only for Full Dhaka)

---

## Sample Results (Full Dhaka)

| Class | Meaning                    | F1-Score (Baseline) | F1-Score (Weighted) |
|-------|----------------------------|----------------------|---------------------|
| `0`   | Local roads (residential)  | 0.89                 | 0.50                |
| `1`   | Secondary/Tertiary roads   | 0.08                 | 0.20                |
| `2`   | Primary roads              | 0.00                 | 0.00                |
| `3`   | Trunk/Motorway             | 0.00                 | 0.07                |

*Due to extreme class imbalance in real-world OSM tags, the baseline model was biased toward the majority class (`residential`). After applying class-weighted loss, the model began to learn patterns for rare classes like secondary and trunk roads. Performance can be further improved with feature scaling, better GNN architectures, and richer input features.*
---

## Future Work

- Use GCN, GraphSAGE, or GAT architectures
- Normalize features and expand input dimensions
- Add road curvature, degree centrality, or clustering coefficient
- Use satellite or traffic data for richer features
- Generalize to Bangladesh-wide road network
