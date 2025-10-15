
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# CSA-DGCN

CSA-DGCN is a **Channel-wise Self-Attention Dynamic Graph Convolutional Network** designed for aircraft landing gear load prediction using FBG (Fiber Bragg Grating) strain measurements.
It models both **sensor topology** and **channel-level dependencies**, dynamically learning structural relationships to improve multi-directional load prediction accuracy across different buffer stroke conditions.
The framework integrates graph convolution, attention fusion, and stroke-aware load regression modules in a unified architecture.

> **Note:** Due to data confidentiality, original datasets and calibration results are **not included** in this repository. Users can prepare their own data following the feeder interface.

---

## 🚀 Features

-- Channel-wise graph convolution for local structural feature learning
-- Self-attention mechanism to enhance inter-channel dependency modeling
-- Dynamic topology updating via adaptive adjacency matrices
-- High-precision multi-directional load prediction ($P_x$, $P_y$, $P_z$)
-- Modular and extensible PyTorch implementation for further research

---

## 📁 Project Structure

```
CSA-DGCN/
├── config/             # Configuration files (hyperparameters, paths)
├── feeders/            # Data loading and preprocessing
├── graph/              # Graph structure definition and builders
├── model/              # CSA-DGCN model and submodules (CGA Block, etc.)
├── src/
│   ├── args.py         # Command-line and configuration parsing
│   ├── trainer.py      # Training and evaluation pipeline
│   ├── utils.py        # Utility functions (metrics, plotting, etc.)
│   └── fig.py          # Visualization and result plotting
├── tools/
│   └── model_analysis.py  # Model structure and performance analysis tools
├── main.py             # Main entry for training and testing
├── requirements.txt    # Python package dependencies
├── LICENSE             # License file
└── .gitignore          # Ignored files and folders
```

---

## 🛠️ How to Run

1. **Clone this repository**

   ```bash
   git clone https://github.com/Mu-Tang/CSA-DGCN.git
   cd CSA-DGCN
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**

   * Place your strain measurement files under `data/`
   * Follow the expected format described in `feeders/` (input: strain sequences; output: load vectors `[x, y, z]`)

4. **Train or evaluate the model**

   ```bash
   python main.py --cfg config/csa_dgcn.yaml
   ```

   Trained weights, logs, and visualizations will be saved automatically to the `works/` folder.

---

## 📦 Requirements

* Python >= 3.8
* PyTorch 2.4.x (CUDA 11.8)
* torch-geometric 2.6.1
* numpy, pandas, matplotlib, seaborn, scikit-learn

(You can install everything directly via `requirements.txt`.)

---

## 📈 Example Output

When executed successfully, the system will generate:

* Training logs and metrics (MAPE, NRMSE, R²)
* Predicted vs. Ground Truth load plots for X/Y/Z directions
* Error distribution and regression curve visualization
* Saved model weights for each cross-validation fold

> All result figures will be saved as `.png` files with publication-ready resolution.

---

## ⚡ Notes

* Datasets and experimental results are **not included**.
* Modify parameters or paths via `config/` or `args.py`.
* Pretrained weights are not released due to proprietary data constraints.
* The model supports both **single-fold** and **cross-validation** training strategies.

---

## 📜 License

This project is released under the **MIT License**.
You are free to use, modify, and distribute it under the same license terms.

---

## 🙋‍♂️ Contact

For any questions or discussions, please open an issue or contact the authors via GitHub:
👉 [https://github.com/Mu-Tang/CSA-DGCN](https://github.com/Mu-Tang/CSA-DGCN)

---
