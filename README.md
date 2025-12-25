# GMB-KAN: Gated Multi-Basis Kolmogorov-Arnold Networks for Time Series Anomaly Detection

Official implementation of the paper: **"GMB-KAN: Gated Multi-Basis Kolmogorov-Arnold Networks for Time Series Anomaly Detection"**.

GMB-KAN is a physics-informed architecture that adaptively unifies multiple domain-specific experts (Wavelet, Chebyshev, and Fourier bases) through a learnable gating mechanism to resolve the fundamental "Forecasting-Reconstruction Dilemma" in Time Series Anomaly Detection (TSAD).

## 🌟 Key Contributions

- **Novel Multi-Basis Architecture**: Introduces a Gated Multi-Basis framework that achieves precise time-frequency localization by adaptively integrating DWT, Chebyshev, and Fourier bases.

- **Resolution of the TSAD Dilemma**: Theoretically and empirically addresses the "identity mapping" issue in forecasting models and the "underfitting" limitations in reconstruction paradigms.

- **SOTA Performance**: Specifically, GMB-KAN achieves a **0.9132 PA-AUPRC** on the UCR dataset, representing an **11.5% relative gain** over the previous best-in-class method, KAN-AD.

- **Robustness in Heterogeneous Environments**: Demonstrates superior discriminative power across diverse and complex real-world signal distributions.

  ------

  

## 🛠️ Environment Setup

We recommend using `Conda` to manage your environment. The code is tested with Python 3.9 and PyTorch 1.12+.

1. Create Environment

   ```
   conda create -n gmbkan python=3.9 -y
   conda activate gmbkan
   ```

2. Install Dependencies

   ```
   pip install -r requirements.txt
   ```

------



## 🚀 Getting Started

### 1. Data Preparation

Place your datasets in the `data/` directory. The model supports standard benchmarks including **UCR**, **SWaT**, **WADI**, and **SMD**.

### 2. Training & Evaluation

To train GMB-KAN and evaluate it on the UCR benchmark, run:

```
python run.py
```

------



## Experimental Results

Empirically, GMB-KAN sets a new performance frontier. The following table highlights the performance (PA-AUPRC) on the UCR benchmark:

|         **Method**         | **UCR (Avg)** | **Improvement (Rel.)** |
| :------------------------: | :-----------: | :--------------------: |
|       Transformer-AD       |    0.7241     |           -            |
|         VAE-based          |    0.7812     |           -            |
| KAN-AD (Zhou et al., 2024) |    0.8188     |        Baseline        |
|     **GMB-KAN (Ours)**     |  **0.9132**   |       **+11.5%**       |

------



Here is a professional, academic-grade **README.md** file in English, tailored for your **GMB-KAN** project. It incorporates the refined terminology we developed (e.g., "Heterogeneous," "Relative gain," "Spectral Disentanglement") to ensure it meets the standards of top-tier conferences like ICML.

------

# GMB-KAN: Gated Multi-Basis Kolmogorov-Arnold Networks for Time Series Anomaly Detection

Official implementation of the paper: **"GMB-KAN: Gated Multi-Basis Kolmogorov-Arnold Networks for Time Series Anomaly Detection"**.

GMB-KAN is a physics-informed architecture that adaptively unifies multiple domain-specific experts (Wavelet, Chebyshev, and Fourier bases) through a learnable gating mechanism to resolve the fundamental "Forecasting-Reconstruction Dilemma" in Time Series Anomaly Detection (TSAD).

------

## 🌟 Key Contributions

- **Novel Multi-Basis Architecture**: Introduces a Gated Multi-Basis framework that achieves precise time-frequency localization by adaptively integrating DWT, Chebyshev, and Fourier bases.
- **Resolution of the TSAD Dilemma**: Theoretically and empirically addresses the "identity mapping" issue in forecasting models and the "underfitting" limitations in reconstruction paradigms.
- **SOTA Performance**: Specifically, GMB-KAN achieves a **0.9132 PA-AUPRC** on the UCR dataset, representing an **11.5% relative gain** over the previous best-in-class method, KAN-AD.
- **Robustness in Heterogeneous Environments**: Demonstrates superior discriminative power across diverse and complex real-world signal distributions.

------

## 🛠️ Environment Setup

We recommend using `Conda` to manage your environment. The code is tested with Python 3.9 and PyTorch 1.12+.

### 1. Create Environment

Bash

```
conda create -n gmbkan python=3.9 -y
conda activate gmbkan
```

### 2. Install Dependencies

Bash

```
pip install -r requirements.txt
```

#### `requirements.txt`

Plaintext

```
torch>=1.12.0
numpy>=1.21.0
tqdm
torchinfo
pandas
scikit-learn
matplotlib
```

------

## 🚀 Getting Started

### 1. Data Preparation

Place your datasets in the `data/` directory. The model supports standard benchmarks including **UCR**, **SWaT**, **WADI**, and **SMD**.

### 2. Training & Evaluation

To train GMB-KAN and evaluate it on the UCR benchmark, run:

Bash

```
python main.py --dataset UCR --window 100 --order 3 --lr 0.001 --batch_size 64
```

### 3. Hyperparameters

- `--window`: Length of the input sliding window.
- `--order`: Depth of DWT decomposition and degree of Chebyshev polynomials.
- `--lr`: Learning rate (Adam optimizer).

------

## 📊 Experimental Results

Empirically, GMB-KAN sets a new performance frontier. The following table highlights the performance (PA-AUPRC) on the UCR benchmark:

| **Method**                 | **UCR (Avg)** | **Improvement (Rel.)** |
| -------------------------- | ------------- | ---------------------- |
| Transformer-AD             | 0.7241        | -                      |
| VAE-based                  | 0.7812        | -                      |
| KAN-AD (Zhou et al., 2024) | 0.8188        | Baseline               |
| **GMB-KAN (Ours)**         | **0.9132**    | **+11.5%**             |

------

## 📂 Project Structure

```
.
├── GMB_KAN/             # GMB-KAN implementation
│   └── GMB_KAN.py      # Core logic for Gating and Multi-Basis experts
├── dataset/               # Dataset directory
├── run.py             # Entry point for training and testing
├── requirements.txt    # Environment dependencies
└── README.md           # Documentation
```

------

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
