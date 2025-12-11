# Patient Clustering and Risk Profiling Using Machine Learning  

---

## Problem Description

Healthcare systems often struggle to identify groups of patients with similar risk profiles. Wide variation in demographics, lifestyle behaviors, chronic conditions, and medical utilization makes traditional risk-scoring approaches incomplete or inaccurate.

The goal of this project was to apply modern machine learning techniques to:

1. Identify natural patient segments based on observed characteristics  
2. Profile and compare these patient segments  
3. Build a predictive model capable of assigning new individuals to these clusters  

This approach provides a structured framework for population health analytics, resource allocation, and management of high-risk patient groups.

---

## Dataset

The project uses a publicly available dataset from Kaggle containing:

- Over **100,000 patient records**  
- **54 demographic, socioeconomic, lifestyle, and clinical features**

Dataset link:  
https://www.kaggle.com/datasets/mohankrishnathalla/medical-insurance-cost-prediction

Below is an example preview of the encoded dataset:

![Dataset](vizualizations/EDA/Dataset%20header.png)

---

## Repository Structure
```
Patient-Clustering-Project/
│
├── data/ # Raw and processed datasets
│ ├── medical_insurance.csv
│ ├── cluster_profiles_full.csv
│
├── notebooks/ # Jupyter notebooks for analysis and modeling
│ ├── Insurance Cost & Risk Analysis.ipynb
│ └── Patient Clustering.ipynb
│
├── vizualizations/ # All project images and graphics
│ ├── EDA/ # Exploratory data analysis plots
│ └── Results/ # Model outputs (ROC curves, confusion matrices, etc.)
│
├── LICENSE
└── README.md
```

## Data Cleaning and Preprocessing

Preprocessing included the following steps:

- Imputation of missing values  
- Standardization of numeric features  
- One-hot encoding of categorical variables  

After preprocessing, the dataset expanded to over **80 engineered features**.

Because this dimensionality is too large for K-Means clustering, we used a correlation matrix to identify the most relevant variables for segmentation.

---

## Phase 1 — Unsupervised Learning (K-Means Clustering)

### Feature Selection  
We selected key demographic, lifestyle, clinical, and chronic condition variables based on:

- Correlation analysis  
- Avoidance of multicollinearity  
- Clinical interpretability  

### Determining Optimal K  
K-Means clustering was applied, and an elbow plot was used to select **k = 4** as the optimal number of clusters.

![Elbow Plot](vizualizations/Results/Elbow%20Method%20.png)

### Cluster Profiling  
After clustering, each group was profiled using:

- Chronic disease burden  
- BMI, blood pressure, LDL, HbA1c  
- Lifestyle behaviors (smoking, alcohol use)  
- Annual medical cost  
- General healthcare utilization  

These profiles allowed us to interpret each cluster as a distinct patient type.

---

## Phase 2 — Supervised Learning (Predicting Cluster Labels)

To predict cluster membership for new patients and validate cluster separability, we trained a **Random Forest classifier**.

### Model Details

- Same preprocessing pipeline as clustering  
- 80/20 stratified train-test split  
- 250 trees, minimum leaf size of 5  
- Balanced class weighting  
- Evaluation using:
  - Confusion matrix  
  - Classification report  
  - Multiclass ROC curves  
  - AUC scores  

![Confusion Matrix](vizualizations/Results/Confusion_Matrix.png)

### Model Performance

- **Accuracy:** 0.9673  
- ROC curves for each cluster displayed **AUC values > 0.99**, indicating excellent separability between patient groups.

![ROC Curves](vizualizations/Results/ROC%20Curves.png)
---

## Findings

### Key Insights

- **Chronic disease burden outweighs age** as a predictor of medical risk.  
  Older adults without chronic conditions had lower risk profiles than younger individuals with at least one chronic disease.

- **Sex differences in annual cost were negligible.**

- **BMI alone is not a strong predictor of cost** when age and chronic diseases are accounted for.

- **Income had minimal effect** on medical expenditure.  
  This may be due to higher-income individuals having better insurance coverage.

- The Random Forest model's strong performance (**accuracy = 0.9673**) validates the robustness of the cluster groups.

### Important Predictors Identified

- Chronic diseases  
- Blood pressure  
- BMI  
- LDL and HbA1c levels  
- Smoking and alcohol use  
- Age  
![Features](vizualizations/Results/Top%2020%20Features.png)
---

## Work Division Among Team Members

All team members contributed equally to:

- Problem definition and refinement  
- Design of the clustering and modeling approach  
- Execution of data analysis and model evaluation  
- Preparation of the presentation and written report  

Responsibilities were shared collaboratively throughout the project.

---

## Use of Libraries and Methods

All work was completed using standard Python libraries:

- pandas  
- NumPy  
- scikit-learn  
- matplotlib  
- seaborn  

Additional scikit-learn components used:

- `StandardScaler`  
- `OneHotEncoder`  
- `SimpleImputer`  
- `ColumnTransformer`  
- `Pipeline`

These tools ensured clean, efficient preprocessing and modeling pipelines.

No external advanced machine learning packages were used.

---

## Use of Open-Source Code

To assist with exploratory analysis, we used a pre-made correlation matrix by **Lighton N. Kalumba** from a Kaggle notebook:

**“Health Cost Prediction ML 95% + Full Analysis”**

This was used strictly to help identify the most relevant features for clustering.  
All other modeling was developed independently.

---

## Citations

Author(s). Title of Article. Journal Name. Year;Volume(Issue):Pages. https://pmc.ncbi.nlm.nih.gov/articles/PMC9180316/  

Kaiser Family Foundation. (2025, October 8). *Health care costs and affordability.* https://www.kff.org/health-costs/health-policy-101-health-care-costs-and-affordability/  

Centers for Disease Control and Prevention. (2025, January 28). *High blood pressure facts.* https://www.cdc.gov/high-blood-pressure/data-research/facts-stats/index.html  

National Library of Medicine. (2025, May 20). *Hemoglobin A1C (HbA1c) test.* https://medlineplus.gov/lab-tests/hemoglobin-a1c-hba1c-test/  

American Heart Association. (2023). *HDL (Good), LDL (Bad) Cholesterol and Triglycerides.* https://www.heart.org/en/health-topics/cholesterol/hdl-good-ldl-bad-cholesterol-and-triglycerides  

