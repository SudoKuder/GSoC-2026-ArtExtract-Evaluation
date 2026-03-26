# ArtExtract: Artwork Analysis & Retrieval

This repository contains two machine learning projects focused on the deep understanding, classification, and retrieval of fine art images. 

## 🎨 Task 1: Multi-Task Art Classification (CNN-RNN)

**File:** `Task_1_ArtExtract.ipynb`

### Overview
Task 1 implements a multi-task learning pipeline to simultaneously predict the **Genre, Style, and Artist** of a given painting. Instead of predicting these independently, the model uses a sequential dependency approach (Genre $\rightarrow$ Style $\rightarrow$ Artist) to leverage the natural relationships between these attributes.

### Architecture
* **Visual Encoder:** A pre-trained **ResNet-50** backbone (with frozen bottom layers) extracts high-level visual features from $224 \times 224$ images.
* **Spatial Attention:** A custom spatial attention module highlights the most relevant regions of the painting for classification.
* **Sequential Decoder:** An **LSTM** network decodes the visual embeddings into predictions. It predicts the Genre first, feeds the Genre embedding into the next LSTM step to predict the Style, and finally uses the Style to predict the Artist.
* **Training Techniques:** Utilizes **Teacher Forcing** during sequence generation to stabilize early training.

### Dataset
* **WikiArt Dataset:** Uses a merged metadata structure mapping image paths to Artist (23 classes), Genre (10 classes), and Style (27 classes) IDs.

---

## 🔍 Task 2: Content-Based Image Retrieval (CBIR)

**File:** `Task_2_ArtExtract(1).pdf` / Notebook

### Overview
Task 2 implements a Siamese-style Content-Based Image Retrieval (CBIR) system. Given a query image of an artwork, the system retrieves the most visually and semantically similar artworks from a massive database in near real-time.

### Architecture & Pipeline
* **Embedding Model (`ArtEmbeddingNet`):** Built on a **ResNet-18** backbone, the network's final fully-connected layer is modified to output a **256-dimensional L2-normalized embedding vector** (an "art fingerprint").
* **Loss Function:** Trained using **Triplet Margin Loss**, which learns to cluster similar artworks closely together in the embedding space while pushing dissimilar artworks apart.
* **Vector Database:** Uses **FAISS** (Facebook AI Similarity Search) to index the generated 256-D vectors, enabling highly scalable and near-instant similarity lookups ($K$-Nearest Neighbors).

### Dataset & Evaluation
* **National Gallery of Art (NGA) Open Data:** Dynamically downloads object metadata and image URLs, fetching high-resolution training crops directly via IIIF image servers.
* **Evaluation Metrics:** The retrieval engine's accuracy is measured using **Precision@K (P@K)** and **Mean Average Precision (mAP)**.

---

## ⚙️ Key Technologies & Dependencies

Both tasks share a common tech stack tailored for deep learning in computer vision:
* **Deep Learning:** `PyTorch`, `torchvision` (ResNet-18, ResNet-50)
* **Data Processing:** `pandas`, `numpy`, `PIL` (Pillow)
* **Vector Search:** `faiss`
* **Visualization:** `matplotlib`, `seaborn`
* **Other:** `requests` (for IIIF image fetching), `scikit-learn`, `scipy`, `tqdm`

## 🚀 Getting Started

1. Clone the repository and install the required dependencies.
2. **Configure your dataset paths:** The notebooks currently contain specific local directory paths (e.g., `./coding/machineLearnning/GSOC_ART/...`). **Please feel free to change these locations in the code to match where you have downloaded and stored the datasets on your own machine.**
3. Ensure the datasets are accessible:
   - **Task 1:** Make sure to update the `repo_base`, `img_dir`, and `base_meta_path` variables in the notebook to point to your local copy of the `WikiArt Dataset`.
   - **Task 2:** This notebook downloads the National Gallery of Art datasets dynamically via API, so just ensure you have an active internet connection to fetch the images from their IIIF servers.
4. Open the respective Jupyter Notebooks and run the cells sequentially to train the models and visualize the results.
