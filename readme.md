The dataset can be reached Through BaiduNet disk (https://pan.baidu.com/s/15HxEI8uaIIQxedATWrjmBQ?pwd=qwdv)

# Unsupervised Graph Embedding and Clustering of IED Networks
This repository contains the code for our project:
> **Unsupervised Graph Embedding and Clustering of Interictal Epileptiform Discharges in MEG Networks to Reveal Spatial-Temporal Subtypes in Focal Epilepsy**  

The goal of the project is to construct source-level MEG functional networks for interictal epileptiform discharges (IEDs), learn low-dimensional graph embeddings using several graph embedding models, and perform unsupervised clustering to identify spatial–temporal IED subtypes in focal epilepsy.

---

## 1. Overall Pipeline
The full processing workflow from raw data to final clustering and statistics is summarized below.
### 1.1 Data acquisition and pre-processing (outside this repo)

These steps are usually performed in clinical / neuroimaging toolboxes and are **not** fully implemented in this codebase, but are listed here for reproducibility.

1. **Data acquisition**
   - Resting-state MEG recorded with 306-channel Elekta Triux system  
     (102 magnetometers, 204 planar gradiometers, sampling rate 1000 Hz).  
   - T1-weighted structural MRI (MPRAGE) for each subject.  

2. **MEG pre-processing & IED identification**
   - Pre-processing implemented in **Brainstorm** (MATLAB):
     - Bad channel detection and repair  
     - tSSS (MaxFilter) for noise suppression  
     - Band-pass filter (1–40 Hz)  
     - SSP to remove eye-movement and cardiac artifacts 
   - IEDs manually marked according to IFCN criteria and confirmed by clinicians;  
     epochs extracted from **–20 ms to +300 ms** around each IED peak.  

3. **MRI segmentation**
   - T1 MRI processed in **FreeSurfer** (`recon-all`) to obtain cortical surfaces and anatomical parcellation (Destrieux atlas, 148 ROIs).   

4. **Source modeling and connectivity**
   - Individual 3-layer BEM head model built in Brainstorm.  
   - sLORETA (or equivalent) used for source reconstruction at cortical vertices, then averaged into 148 ROIs.    
   - For each IED epoch, a **148 × 148 adjacency matrix** is computed using **Phase-Locking Values (PLV)** as connectivity weights (0–1). 
The outputs of this stage are:
- `A`: adjacency matrices for each IED (PLV-based 148×148)
- `deg`: nodal degree (or degree centrality) for each IED
- (optionally) global / nodal graph metrics (aCP, aLP, aEG, aEL, etc.)

In this repository we start from these pre-computed matrices.

---

## 2. Graph Embedding and Clustering Pipeline (code in this repo)

The core of this repository is the **unsupervised graph embedding + clustering** pipeline, implemented in Python.

### Step 1. Data loading

- Load `.mat` / `.npy` files containing:
  - `adjacency` matrices: shape `(N_IED, 148, 148)`
  - `degree` matrices or vectors: shape `(N_IED, 148)` or `(N_IED, 148, 1)`  
- Normalize nodal features (degree centrality) with standard scaling (zero mean, unit variance).  

### Step 2. Graph construction

For each IED:

1. Build a graph `G = (V, E)`:
   - Nodes: 148 cortical ROIs
   - Edges: non-zero entries of adjacency matrix `A`
   - Edge weights: PLV values `A_ij`  

2. Node features:
   - Use normalized **degree centrality** as node features `x_i`. 
These graphs are then fed into different embedding models.

### Step 3. Graph embedding models

We implement four embedding methods:

1. **DeepWalk**
2. **Node2Vec**
   - Both based on random walks + Skip-gram to learn node embeddings.  

3. **GCN (Graph Convolutional Network)**
   - Multi-layer GCN aggregating neighbor information, followed by global pooling to obtain graph-level embeddings.  
4. **GAT (Graph Attention Network)**
   - Multi-head attention over neighbors, then global pooling.  
   - Trained with Adam optimizer for ~300 epochs, minimizing MSE-style loss. 

For GCN/GAT, we use **global mean pooling** to aggregate node embeddings into one vector per graph (per IED).  
The final output of this step is:
- `Z`: graph-level embedding matrix, shape `(N_IED, d_embed)`

### Step 4. Unsupervised clustering

On graph embeddings `Z` we perform:

- **K-Means clustering**
- **Spectral Clustering**

For a range of cluster numbers `K = 2 … 13`, we compute:

- **Silhouette Score (SS)**
- **Davies–Bouldin Index (DBI)**
- **Calinski–Harabasz Index (CHI)**  
In our experiments, the **GAT + K-Means** combination with **K = 4** gave the best performance  
(highest SS, CHI; lowest DBI), and is used as the **default / recommended setup**. 
### Step 5. Network analysis & statistics

Based on clustering assignments:

1. **Global metrics across subtypes**
   - Compare aCP, aLP, aEG, aEL between subtypes using one-way ANOVA + Bonferroni post-hoc tests.  
2. **Nodal metrics & unique nodes**
   - Identify subtype-specific “unique” nodes and edges and visualize them (BrainNet Viewer / Python brain plotting).  
3. **Structure–function correlation**
   - For each subject, compute the frequency of each subtype and correlate with:
     - Age  
     - Cortical thickness of key ROIs (Destrieux) 

4. **Visualization**
   - PCA / t-SNE on graph embeddings  
   - Scatter plots of clusters  
   - Brain network visualizations for each subtype  

---
