# 💠 Visio AI Enterprise Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

> **The efficient, local-first alternative to cloud AI.**  
> Clean, Visualize, and Model your data without writing a single line of code.

---

## 📖 Overview

**Visio AI** is an enterprise-grade Data Science platform designed to democratize Machine Learning. Unlike heavy, resource-intensive Cloud AI solutions, Visio AI runs highly efficient algorithms (XGBoost, Random Forest, etc.) directly on your local hardware.

It serves as a "Command Center" for your data, handling the full pipeline:
1.  **Ingestion & Wrangling** (Cleaning dirty data)
2.  **Exploratory Data Analysis** (Interpreting patterns)
3.  **Predictive Modeling** (Forecasting future trends)
4.  **Computer Vision** (Analyzing images)

---

## ✨ Key Features

### 🏗️ Data Engineering
*   **Universal Loader**: Support for CSV, Excel (`.xlsx`), and Text files.
*   **Smart Wrangling**: 
    *   Auto-detect and fix missing values (Imputation).
    *   One-click "Remove Commas" features for financial datasets.
    *   Type correction (String $\to$ Number).

### 📊 Advanced Visualization
*   **Dual Mode Graphics**: Switch between **Interactive** (Plotly) for exploration and **Static** (Seaborn) for publication.
*   **Smart Suggestions**: The system automatically recommends the right chart (e.g., Heatmap vs Scatter) based on your variables.

### 🧠 Machine Learning Engine
*   **Supervised Learning**: Training interface for **Regression** and **Classification**.
    *   *Algorithms*: XGBoost, Random Forest, JVM, Linear Models, Decision Trees.
*   **Unsupervised Learning**: K-Means Clustering (with 3D Viz) and PCA Dimensionality Reduction.
*   **AutoML**: Automatically trains 6+ models and ranks them on a leaderboard.

### 👁️ Image Intelligence
*   **Multimodal Analysis**: Integrated with **Nvidia Nemotron-12B** for analyzing images.
*   **Tasks**: OCR, Scene Description, Defect Detection.

---

## 🚀 Installation & Usage

### Prerequisites
*   Python 3.8 or higher
*   pip

### Setup
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/StartUp-Jaiho/Visio-AI.git
    cd Visio-AI
    ```

2.  **Create a Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Running the App
```bash
streamlit run Home.py
```
The application will launch automatically in your web browser.

---

## 📂 Project Structure

```text
Visio-AI/
├── Home.py                 # Application Entry Point
├── Docs.html               # Deep Dive Documentation (HTML)
├── Guide.md                # User Guide (Markdown)
├── utils.py                # Shared Utility Functions
├── styles.css              # Enterprise CSS Theme
├── pages/                  # Application Modules
│   ├── 1_Data_Loader.py    # Ingestion & Cleaning
│   ├── 2_EDA.py            # Visualization Engine
│   ├── 3_Supervised.py     # ML Training & Prediction
│   ├── 4_Unsupervised.py   # Clustering & PCA
│   ├── 5_Image_AI.py       # Computer Vision
│   ├── 6_AutoML.py         # Automated Modeling
│   ├── 7_Report.py         # PDF Reporting
│   └── ... (Utilities)
└── assets/                 # Static Assets (Images, Icons)
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ by <strong>Jaiho Labs</strong>
</p>
