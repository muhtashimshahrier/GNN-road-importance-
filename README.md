# Road Segment Importance Prediction using Graph Neural Networks

This project predicts the relative **importance of road segments** in the urban road network of **Dhaka, Bangladesh** using a Graph Neural Network (GNN) pipeline.

A real road network was extracted using **OSMnx**, then processed through a full pipeline:

- **Feature Engineering**: Computed edge-level features like `length` and `betweenness centrality`.
- **Edge Labeling**: Assigned multi-class importance labels using OpenStreetMap `highway` tags.
- **Graph Conversion**: Transformed the graph into **PyTorch Geometric** format (`edge_index`, `edge_attr`, `edge_labels`).
- **Model Training**:
  - Trained a baseline **MLP** using edge features only (no message passing).
  - Trained a **Graph Convolutional Network (GCN)** using `GCNConv` layers to leverage structural information in the graph.

This pipeline was applied to both a small subgraph (Dhanmondi, MLP only) and the full Dhaka city network (~250k edges).

---

## Folder Structure
GNN-road-importance/
├── dhanmondi_test/     # Small-scale prototype
├── dhaka_full/         # Full Dhaka road network (~250k edges)
└── README.md           # Project overview and documentation

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

| Step                 | Description |
|----------------------|-------------|
| Graph Download       | Retrieve road network using OSMnx |
| Edge Labeling        | Assign 4-class importance from OSM `highway` tags |
| Feature Engineering  | Compute edge `length` and `betweenness centrality` |
| PyG Conversion       | Convert NetworkX graph to PyTorch Geometric |
| Model Training       | Train MLP and GCN models to classify edge importance |
| Class Balancing      | Apply class-weighted loss to improve rare class recall |

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
- `train_gnn_gcn.py` (Only for Full Dhaka)

---

## Model Performance Comparison (Full Dhaka)

| Model                | Accuracy | Macro F1 | Notes                                               |
|---------------------|----------|----------|-----------------------------------------------------|
| MLP (Vanilla)       | 0.81     | 0.24     | Overfit to dominant class; poor generalization      |
| MLP (Class-weighted)| 0.35     | 0.19     | Improved minority class recall; lower overall acc.  |
| GCN (GCNConv)       | 0.61     | 0.25     | Balanced performance; better structural learning    |

---

## Interpretation

The baseline MLP achieved high accuracy by favoring the dominant class (`residential`), but failed to generalize to minority classes. Class-weighted loss improved recall for rare classes but significantly reduced overall accuracy due to the lack of structural awareness. In contrast, the GCN model **leveraged road network connectivity** to boost recall and F1 for underrepresented classes — achieving a more balanced performance. This demonstrates the value of graph-based architectures in learning from real-world spatial networks with extreme class imbalance.

---

## Future Work

- Extend to more advanced GNNs like GraphSAGE, GAT, or edge-conditioned convolutions  
- Scale up feature space with normalization, interaction terms, or richer topological descriptors  
- Incorporate geometric features like curvature, angular deviation, and clustering coefficient  
- Integrate external data sources (e.g., satellite imagery, traffic counts, land use) for multimodal learning  
- Generalize the pipeline to a Bangladesh-wide road network and test region-specific transferability  
- Explore self-supervised or contrastive learning for pretraining on unlabeled networks

---

**Note**: Due to file size limits, this repo does not include raw or processed `.graphml` files.


