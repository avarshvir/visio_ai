# 📘 Visio AI: The User Guide

**Visio AI** is a comprehensive, local-first enterprise platform for Data Science and Machine Learning. It enables users to perform end-to-end data analysis—from cleaning dirty CSVs to training XGBoost models—without writing code.

---

## 🚀 Quick Start

1.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the App**:
    ```bash
    streamlit run Home.py
    ```
    *(The app will open in your default browser at `http://localhost:8501`)*

---

## 🛠️ Modules & Features

### 1. Data Loader & Wrangling
*   **Import**: specific support for `.csv`, `.xlsx` (Excel), and `.txt` files.
*   **Auto-Cleaning**: The system can automatically fill missing values (Mean for numbers, Mode for text) with one click.
*   **Wrangling**:
    *   **Remove Commas**: Fixes formatted numbers (e.g., "1,200" → 1200) across the entire dataset.
    *   **Rename/Drop**: standard column operations.
    *   **Type Conversion**: cast strings to numbers or dates.

### 2. Exploratory Data Analysis (EDA)
*   **Smart Suggestions**: The system recommends plots based on your selected data types.
*   **Interactive Plots (Plotly)**: Zoom, pan, and hover over data points.
*   **Static Plots (Seaborn)**: High-quality statistical plots like **Pair Plots** and **Violin Plots**.
*   **Profiling**: View full statistical summaries (Mean, Std Dev, Quantiles) for every column.

### 3. Supervised Machine Learning
Train models to predict outcomes.
*   **Classification** (Predicting Categories like "Spam/Ham"):
    *   Logistic Regression, Random Forest, Decision Tree, SVC, KNN, Naive Bayes, XGBoost.
*   **Regression** (Predicting Numbers like "Price"):
    *   Linear Regression, Random Forest, Decision Tree, SVR, KNN, XGBoost.
*   **Prediction Lab**: After training, you can manually input values to test the model in real-time.

### 4. Unsupervised Learning
Find hidden patterns in unlabeled data.
*   **K-Means Clustering**: Group similar customers or items together. Includes 3D visualization.
*   **PCA**: Dimensionality reduction to visualize complex datasets in 2D or 3D.

### 5. Image AI (Computer Vision)
*   **Model**: Powered by **Nvidia Nemotron-12B-v2**.
*   **Features**: Upload an image and ask questions like "Read the text in this receipt" or "Describe the defect in this product".

### 6. AutoML
*   **Battle Mode**: Runs *all* compatible algorithms on your data simultaneously.
*   **Leaderboard**: Ranks models by Accuracy (or R2 Score) so you know exactly which algorithm performs best.

---

## 💡 Best Practices

*   **Handling Missing Data**: Always check the "Data Loader" first. If your data has `NaN` values, ML models will crash. Use "Auto-Clean" to fix this instantly.
*   **Choosing a Model**:
    *   Start with **Random Forest** or **XGBoost** for the best accuracy.
    *   Use **Logistic/Linear Regression** if you need to explain *why* the model made a decision (interpretability).
*   **Image AI**: For the best results, ensure your specific question is included in the prompt (e.g., "List all items in JSON format").

---

## ❓ FAQ

**Q: Why is my accuracy 100%?**
A: This often happens with small, simple datasets (like Iris). Try a more complex dataset or increase the Test Size split to make it harder for the model.

**Q: Is my data safe?**
A: **Yes.** Visio AI runs locally. Your CSV data never leaves your computer (except for Image AI, which sends the specific image to the Inference API).

**Q: Can I use this offline?**
A: Yes! All tabular data features (EDA, ML, AutoML) work 100% offline. Only Image AI requires an internet connection.
