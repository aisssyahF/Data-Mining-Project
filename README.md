# Data-Mining-Project

# US Accidents Data Mining Project - Group P29
### CDS6314 Project - TRI. 2530

This project analyzes the US Accidents dataset (2016–2023) to perform data preprocessing, pattern discovery, and predictive modeling, culminating in an interactive Streamlit dashboard.

---

## Project Structure
```text
Data Mining Project/
├── 2530_CDS6314_Project_Final_Group_P29.ipynb  # Main analysis notebook
├── app.py                                     # Streamlit interactive dashboard
├── requirements.txt                           # Python dependencies
├── README.md                                  # Project documentation
└── Generated/
    ├── accidents_processed.csv                # Cleaned 500K sample dataset for app
    ├── accident_severity_model.pkl            # Trained Random Forest model
    ├── feature_scaler.pkl                     # StandardScaler for normalization
    ├── model_features.txt                     # List of 20 features used in model
    └── model_metadata.json                    # Model performance metrics
```
---

## 📋 Prerequisites
* **Python**: 3.8 or higher
* **Tools**: Jupyter Notebook / JupyterLab & a Web Browser (to view the Streamlit dashboard)
* **Hardware**: 8GB RAM minimum (16GB recommended) and at least 5GB free disk space.

---

## 📊 Dataset Download
Due to GitHub's file size limits, the dataset must be downloaded manually:
1. **Source**: Visit the [US Accidents Dataset on Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).
2. **File**: Download the `US_Accidents_March23.csv` (~1.5GB compressed / ~4GB uncompressed).
3. **Setup**: Place the CSV file directly in the root project folder (the same folder as the `.ipynb` notebook).

