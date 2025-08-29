# 🧩 HUEM: A High-performance Unified Enhancement Method for Multi- and High-Dimensional Indexes

This repository contains the source code for the paper **HUEM: A High-performance Unified Enhancement Method for Multi- and High-Dimensional Indexes**. It provides implementations for both static and dynamic layout optimization experiments.

🔗 **Resources**: [Paper](https://arxiv) | [Project Page](https://github.com/lizem527/HUEM)

## 📂 Repository Structure

- **HUEM_static_layout_opt/**: Source code for online partial layout reorganization.
  - **dataset/**: Subfolder for data download and management.
  
- **HUEM_dynamic_layout_opt/**: Source code for offline layout optimization.

## 🖥️ Environment

### Single-Node Server

- **OS**: Ubuntu 20.04 (64-bit)
- **CPU**: 2× Intel Xeon Gold 6226 @ 2.90GHz
- **RAM**: 256GB
- **GPU**: NVIDIA RTX A5000
- **Hosted Components**: Relational DB, Vector DB

### Distributed Cluster (3 Nodes)

- Per Node Specs:

  - **OS**: Ubuntu 18.04 (64-bit)
  - **CPU**: Intel Core i7-11700F @ 2.50GHz
  - **RAM**: 16GB
  
- **Hosted Component**: Data Lake

## 🛠️ Static Usage

The `HUEM_static_layout_opt` folder contains experiments for static layout optimization. Static and dynamic experiments are separated: use `HUEM_static_layout_opt` for static experiments and `HUEM_dynamic_layout_opt` for dynamic experiments (see [Dynamic Usage](https://grok.com/chat/12f38543-978d-4244-b876-328566b31215#-dynamic-usage) below).

The `HUEM_static_layout_opt/main.py` script implements the processing pipeline for random data using the **Feature-based Transformation (A) + Gravity-based Movement (G)** approach. This pipeline augments raw data for downstream tasks like clustering and index building. Random data is used by default, but real datasets can be downloaded (see [Data Download](https://grok.com/chat/12f38543-978d-4244-b876-328566b31215#-data-download)).

### 🚀 Quick Run (Example)

```bash
git clone https://github.com/lizem527/HUEM.git
cd HUEM
# Navigate to the static module directory
cd HUEM_static_layout_opt

# Run the static layout generation (A+G processing for random data)
python main.py
```

### 📥 Data Download

To download the reference dataset from within `HUEM_static_layout_opt`:

```bash
cd dataset
python dataset_download.py
```

## 🔄 Dynamic Usage

To work with dynamic layout optimization, navigate to the dynamic module:

```bash
cd HUEM_dynamic_layout_opt
```

### ⚙️ Configuration

- **Data**: `resources/config/*.json`
- **Query**: `resources/config/*.p` (pickle files with predicates for each query)
- **Overall**: `resources/params/*.json`

### 🔍 Methods of Comparison

To run different methods:

- **Generate candidate data layouts**:

  ```bash
  python HUEM_layout_main.py --config config_name
  ```

- **Modified HUEM/OREO algorithm** (use `offline/dis.py` for OREO, `offline/new_dis.py` for HUEM):

  ```bash
  python HUEM_OREO_main.py --config config_name
  ```

- **Static (Original)**:

  ```bash
  python HUEM_Original_main.py --config config_name
  ```

- **Measure end-to-end time**:

  ```bash
  python HUEM_replay_main.py --config config_name --rewrite --root /path/to/partition --alg random
  ```

## 📧 Contact Us

For any issues or questions, please contact the code maintainer at [lizeming@bit.edu.cn](mailto:lizeming@bit.edu.cn).
