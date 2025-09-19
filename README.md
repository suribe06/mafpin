# MAFPIN: Matrix Factorization with Properties of Inferred Networks

## Overview

MAFPIN is a research project that aims to **improve collaborative filtering recommendation systems** by incorporating user network properties derived from cascade data. The core hypothesis is that user centrality metrics in social influence networks can enhance traditional matrix factorization algorithms to provide better recommendations.

### Key Innovation

Traditional collaborative filtering relies solely on user-item rating patterns. MAFPIN enhances this approach by:

1. **Inferring social influence networks** from temporal cascade data using the NetInf algorithm
2. **Computing user centrality metrics** that capture each user's position and influence in the network
3. **Integrating these metrics as user attributes** in Collaborative Matrix Factorization (CMF) models
4. **Evaluating performance improvements** in recommendation accuracy (RMSE reduction)

The project implements a complete pipeline from cascade generation to enhanced recommendations, providing insights into how social network structure can inform recommendation systems.

## Installation

### Requirements

- **Python 3.9** (maximum version - compatibility requirement for some dependencies)
- Git for cloning the repository

### Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/suribe06/mafpin.git
cd mafpin
```

2. **Create a virtual environment (recommended):**

```bash
python -m venv mafpin_env
source mafpin_env/bin/activate  # On Windows: mafpin_env\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

### Key Dependencies

- `cmfrec`: Collaborative Matrix Factorization library
- `snap-stanford`: Network analysis (SNAP-py)
- `pandas`, `numpy`, `scipy`: Data processing and numerical computing
- `scikit-learn`: Machine learning utilities
- `tqdm`: Progress bars for long-running processes

## Methodology

### 1. Cascade Generation

**Purpose:** Convert temporal rating data into cascade format suitable for network inference.

A **cascade** represents the temporal sequence of user interactions with a specific item (e.g., movie ratings). The cascade format captures the "viral" or influence-based spread of preferences through a user network.

**Input Format:** CSV file with columns:

- `UserId`: Unique user identifier
- `ItemId`: Unique item identifier
- `Rating`: Rating value (e.g., 1-5 stars)
- `timestamp`: Unix timestamp of the interaction

**Output Format:** `cascades.txt` file containing:

- User ID mappings
- Cascade data (one line per item): `user1,time1,user2,time2,...`

### 2. Network Inference with NetInf

**NetInf Algorithm:**  
NetInf is a probabilistic algorithm that infers influence / diffusion networks from cascade data. It models how information, preferences, or influence propagate through a network by analyzing temporal patterns in cascades.

**Core Concept:**  
If user A influences user B, then A’s interactions should tend to occur before B’s interactions across multiple cascades. NetInf quantifies these temporal dependencies to reconstruct the underlying influence network.

**Mechanics & Guarantees:**  
- Assumes a **static directed network** and that cascades spread via directed trees.
- Because inferring the exact network is NP-hard, NetInf uses a **greedy submodular optimization** that achieves a provable near-optimal approximation.
- Requires specifying **k**, the number of edges to include in the inferred network. NetInf selects the top-k edges that maximize likelihood over all cascades.

**Available Models:**  
- **Exponential Model (0):** Transmission probability decays exponentially with time  
- **Power Law Model (1):** Heavy-tailed transmission probability distribution  
- **Rayleigh Model (2):** Bell-shaped transmission probability with a modal delay

**Alpha Parameter (α):** Controls the transmission rate/speed in the diffusion models:

- **Higher α:** Faster information transmission, shorter delays
- **Lower α:** Slower transmission, longer delays

### 2.1 Alpha Search Space Calculation

The algorithm automatically computes model-specific α ranges using cascade temporal statistics.

#### 2.1.1. Median Delta (Δ)
We first compute the **median time difference** between user interactions within cascades:

$$
\tilde{\Delta} = \text{median}\lbrace t_i - t_j \mid t_i > t_j \rbrace
$$

Timestamps are assumed to be in **Unix epoch seconds**, but they may be converted to **days** or **years** as long as the same unit system is used consistently for Δ, α, and cascade timestamps.

---

#### 2.1.2. Model-Specific Centers

Each transmission model has a theoretical relationship between its **median** and **α**, which defines a natural center point:

- **Exponential**
<div align="center">

$$
m = \frac{\ln(2)}{\alpha} \quad \Rightarrow \quad \alpha_{\text{center}} = \frac{\ln(2)}{\tilde{\Delta}}
$$

</div>

- **Rayleigh**
<div align="center">

$$
m = \sqrt{\frac{2 \ln(2)}{\alpha}} \quad \Rightarrow \quad \alpha_{\text{center}} = \frac{2 \ln(2)}{\tilde{\Delta}^2}
$$

</div>

- **Power-law**
<div align="center">

$$
m = 2^{1/\alpha}\Delta_{\min} \quad \Rightarrow \quad 
\alpha = \frac{\ln(2)}{\ln(\tilde{\Delta}/\Delta_{\min})}
$$

</div>
However, for typical datasets this value falls below 1, while the inference algorithm only supports
<div align="center">

$$\alpha \geq 1.$$

</div>
Therefore, in practice we use a fixed linear grid in the range:
<div align="center">

$$
\alpha \in [1, 3] \quad \text{or} \quad [1, 5].
$$

</div>

---

#### 2.1.3. Log-Scale Grid (Exponential & Rayleigh)

For Exponential and Rayleigh models, we explore α values around the center on a **logarithmic scale** to capture both slower and faster transmission rates:

- **Range factor**: \( r \) (typical values: 10–100)  
- **Grid size**: \( N \) (typical values: 20–50)  

The grid formula is:

$$
\alpha_i = \alpha_{\text{center}} \cdot r^{\left( \tfrac{2i}{N-1}-1 \right)}, 
\quad i = 0, 1, \dots, N-1
$$

This ensures exactly half of the values are smaller than the center and half are larger, evenly spaced in log-space. **IMPORTANT**: Always ensure the same unit system is used for **Δ, α, and timestamps**.

---

### 3. Centrality Metrics Calculation

After network inference, the system calculates comprehensive centrality metrics for each user:

**Computed Metrics:**

- **Degree Centrality:** Number of direct connections (local influence)
- **Betweenness Centrality:** Control over information flow between other users
- **Closeness Centrality:** Ability to quickly reach all other users in the network
- **Eigenvector Centrality:** Influence based on connections to other influential users
- **PageRank:** Google's algorithm adapted for influence networks
- **Clustering Coefficient:** Tendency to form tightly-knit groups
- **Eccentricity:** Maximum distance to any other user (network periphery measure)

**Human Interpretation:**

- **High Degree:** "Social connector" - many direct relationships
- **High Betweenness:** "Bridge builder" - connects different communities
- **High Closeness:** "Information broker" - quickly spreads/receives information
- **High Eigenvector/PageRank:** "Influencer" - connected to other important users
- **High Clustering:** "Community member" - part of tight-knit groups
- **Low Eccentricity:** "Network center" - close to all other users

### 4. Enhanced Matrix Factorization

**Baseline CMF:** Standard Collaborative Matrix Factorization using only user-item rating patterns.

**Enhanced CMF:** Incorporates user centrality metrics as additional user attributes using the `cmfrec` library's user attributes feature (`U` parameter).

**Process:**

1. **Hyperparameter Search:** Find optimal `k` (latent factors) and `λ` (regularization) using cross-validation
2. **Baseline Evaluation:** Train CMF with only rating data, measure RMSE
3. **Enhanced Evaluation:** Train CMF with ratings + centrality attributes, measure RMSE
4. **Improvement Analysis:** Compare RMSE values to quantify enhancement

**Success Metrics:**

- **RMSE Reduction:** Lower RMSE indicates better prediction accuracy
- **Improvement Percentage:** `((baseline_RMSE - enhanced_RMSE) / baseline_RMSE) * 100`
- **Consistency:** Positive improvements across multiple network models

## Usage Instructions

### Directory Structure

```
mafpin/
├── data/                           # Data files and outputs
│   ├── ratings_small.csv          # Input rating dataset
│   ├── cascades.txt               # Generated cascades
│   ├── inferred_networks/         # NetInf output networks
│   └── centrality_metrics/        # Computed centrality data
├── networks/                      # Network analysis pipeline
├── matrix_factorization/          # CMF implementation
├── plots/                         # Generated visualizations
└── requirements.txt
```

### Execution Pipeline

#### Step 1: Generate Cascades

**Execute as Python script:**

```bash
cd networks
python generate_cascades.py ratings_small
```

**Supports command-line arguments:**

```bash
python generate_cascades.py --help                    # Show help
python generate_cascades.py                          # List available datasets
python generate_cascades.py [dataset_name]           # Generate cascades
```

#### Step 2: Network Inference

**Execute with command-line interface:**

```bash
cd networks
python run_inference.py --help                    # Show help
python run_inference.py --model 0 --N 100 --r 100 # Single model
python run_inference.py --all-models --N 50       # All three models
```

**Available parameters:**

- `--cascades`: Cascades file name (default: cascades.txt)
- `--N`: Number of α values in log-scale grid (default: 100)
- `--model`: Model type (0=exponential, 1=powerlaw, 2=rayleigh)
- `--max-iter`: Maximum NetInf iterations (default: 2000)
- `--r`: Range factor for α grid computation (default: 100.0)
- `--all-models`: Run inference for all three models

**Alternative (modify parameters in script):**

```bash
cd networks
python infer_networks.py
```

#### Step 3: Calculate Centrality Metrics

**Execute with command-line interface:**

```bash
cd networks
python run_centrality_metrics.py --help           # Show help
python run_centrality_metrics.py --all-models      # All models
python run_centrality_metrics.py --model exponential --plots # Single model with plots
python run_centrality_metrics.py --single-network network_file.txt # Single network
```

**Available parameters:**

- `--single-network`: Process a single network file
- `--all-models`: Process all networks from all models
- `--model`: Process specific model (exponential, powerlaw, rayleigh)
- `--plots`: Generate visualization plots

**Alternative (modify parameters in script):**

```bash
cd networks
python calculate_centrality_metrics.py
```

#### Step 4: Enhanced Matrix Factorization

**Standard CMF evaluation:**

```bash
cd matrix_factorization
python cmf.py
```

**Centrality-enhanced evaluation:**

```bash
cd matrix_factorization
python cmf_centrality.py
```

**Shell command support:** Both support `python script.py` execution only

### File Execution Types

#### Command-Line Interface Support:

- `networks/generate_cascades.py` ✓ (with CLI arguments)
- `networks/run_inference.py` ✓ (with CLI arguments)
- `networks/run_centrality_metrics.py` ✓ (with CLI arguments)

#### Python Script Only:

- `networks/infer_networks.py`
- `networks/calculate_centrality_metrics.py`
- `matrix_factorization/cmf.py`
- `matrix_factorization/cmf_centrality.py`

### Expected Outputs

1. **Cascades:** `data/cascades.txt` with temporal interaction sequences
2. **Networks:** `data/inferred_networks/{model}/` with inferred influence networks
3. **Centrality Metrics:** `data/centrality_metrics/{model}/` with user centrality CSV files
4. **Evaluation Results:** Console output with RMSE comparisons and improvement percentages
5. **Visualizations:** `plots/` directory with analysis charts

### Performance Expectations

- **Cascade Generation:** Minutes for small datasets (<100K ratings)
- **Network Inference:** Hours for 100 networks × 3 models (highly dependent on data size)
- **Centrality Calculation:** Minutes to hours depending on network size
- **CMF Evaluation:** Minutes for hyperparameter search and evaluation

## Research Context

This project implements and extends network-based collaborative filtering research, specifically investigating whether social influence networks derived from temporal patterns can enhance traditional matrix factorization approaches. The pipeline provides a complete framework for testing this hypothesis on real-world rating datasets.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{doi:10.1142/S1793830923500520,
author = {Uribe, Santiago and Ramirez, Carlos and Finke, Jorge},
title = {Recommender systems based on matrix factorization and the properties of inferred social networks},
journal = {Discrete Mathematics, Algorithms and Applications},
volume = {16},
number = {05},
pages = {2350052},
year = {2024},
doi = {10.1142/S1793830923500520},
URL = {https://doi.org/10.1142/S1793830923500520}
}
```

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

This is a research project. For questions or contributions, please refer to the repository maintainer.
