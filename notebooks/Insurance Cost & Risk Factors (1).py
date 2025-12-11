# Patient Clustering (Unsupervised) to find natural segments based on demographics, lifestyle, and medical history
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# load dataset
df = pd.read_csv(
    "data/medical_insurance.csv",
    encoding = 'utf-8'
)

# handle missing values to prevent errors
df.replace(["", "NA", "N/A", "NaN", "None"], np.nan, inplace=True)

# print the dataset dimensions to see it loaded right
print("Data shape:", df.shape)


# these are our features for clustering
cluster_features = [
    # demographics
    "age", "sex",

    # personal lifestyle
    "bmi", "smoker", "alcohol_freq",

    # bp, LDL and glucoseblood sugar
    "systolic_bp", "diastolic_bp", "ldl", "hba1c",

    # amount of chronic conditions
    "chronic_count"
]


# make sure features exist ing original df
cluster_features = [c for c in cluster_features if c in df.columns]

# create clustering variables df
X = df[cluster_features].copy()

# separate categorical vs numeric (for pipeline)
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

# preprocessing pipeline
# for numerical
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), # replace missing values with the median
    ("scaler", StandardScaler()) # k means uses distance so we need to scale 
])

# for categorical
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")), # replace mmissing values with majority 
    ("ohe", OneHotEncoder(handle_unknown="ignore")) # binarize
])

# output numeric matrix based on above preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ("cat", cat_pipe, categorical_cols),
        ("num", num_pipe, numeric_cols)
    ]
)

# fit & transform to get matrix for clustering
X_pre = preprocess.fit_transform(X)
print("\nTransformed feature matrix:", X_pre.shape)

# elbow method to choose k
inertia = [] # list to store inertia values (how far points are cluster center)
k_range = range(2, 11) # range for cluster counts

# try k means (2 to 10) for each k, fit the model and record the inertia
for k in k_range:
    km_tmp = KMeans(n_clusters=k, random_state=100, n_init=10)
    km_tmp.fit(X_pre)
    inertia.append(km_tmp.inertia_)

# use inertia to build elbow plot
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker="o")
plt.xticks(k_range)
plt.xlabel("(k) clusters")
plt.ylabel("Total in cluster variation")
plt.title("Elbow Method for Patient Clusters")
plt.show()

# fit final k means model
k_chose = 4 
kmeans = KMeans(n_clusters=k_chose, random_state=100, n_init=20)
labels = kmeans.fit_predict(X_pre)

df["cluster"] = labels

print("\nCluster sizes:")
print(df["cluster"].value_counts().sort_index())

# calculate in cluster averages
# (We can still look at cost & utilization by cluster, even though they weren't used to form the clusters)
profile_vars = cluster_features.copy()

# add cost & utilization (for interpretation only)
for extra in [
    "annual_medical_cost",
    "visits_last_year", "hospitalizations_last_3yrs",
    "days_hospitalized_last_3yrs", "medication_count"
]:
    if extra in df.columns:
        profile_vars.append(extra)

# only keep numeric columns when computing means
cluster_profile = (
    df[profile_vars + ["cluster"]]
    .select_dtypes(include="number")
    .groupby("cluster")
    .mean()
    .round(2)
)

print(cluster_profile)

cluster_profile.to_csv("cluster_profiles_full.csv")
print("see saved csv file!!!")

# cost breakdown by cluster
if "annual_medical_cost" in df.columns:
    cost_by_cluster = (
        df.groupby("cluster")["annual_medical_cost"]
          .agg(["count", "mean", "sum"])
          .rename(columns={"count": "n", "mean": "avg_cost", "sum": "total_cost"})
    )
    cost_by_cluster["cost_share_%"] = (
        100 * cost_by_cluster["total_cost"] / cost_by_cluster["total_cost"].sum()
    )
    print("\nCost by cluster:")
    print(cost_by_cluster.round(2))

# visuals
# cluster level cost comparison barplot
plt.figure(figsize=(8,5))
plt.bar(cost_by_cluster.index, cost_by_cluster["avg_cost"])
plt.xlabel("Cluster")
plt.ylabel("Average Annual Medical Cost")
plt.title("Average Medical Cost by Cluster")
plt.show()

# clsuter size barplot
plt.figure(figsize=(8,5))
plt.bar(df["cluster"].value_counts().index,
        df["cluster"].value_counts().values)
plt.xlabel("Cluster")
plt.ylabel("Number of Patients")
plt.title("Cluster Sizes")
plt.show()

# boxplots to compare how each risk factor is distributed across the patient clusters 
box_vars = ["bmi", "chronic_count", "hba1c", "visits_last_year", "medication_count", "systolic_bp", "diastolic_bp"]

for var in box_vars:
    plt.figure(figsize=(7, 4))
    df.boxplot(column=var, by="cluster")
    plt.title(f"{var} by Cluster")
    plt.suptitle("")  # fix pandas title
    plt.xticks(ticks=[1, 2, 3, 4], labels=["1", "2", "3", "4"]) # I added this to allign with our cluster numberings (to start at 1 not 0)
    plt.xlabel("Cluster")
    plt.ylabel(var)
    plt.tight_layout()
    plt.show()


# random forest

print("\nRandom forest below")

# same predictors used for clustering
X_rf = df[cluster_features].copy()

# y = cluster labels from k means
y_rf = df["cluster"]

# test / train split
X_train, X_test, y_train, y_test = train_test_split(
    X_rf, y_rf,
    test_size=0.20,
    random_state=100,
    stratify=y_rf
)

# build pipeline
rf_med = Pipeline(steps=[
    ("prep", preprocess),
    ("rf", RandomForestClassifier(
        n_estimators=250, # 250 trees
        min_samples_leaf=5, 
        class_weight="balanced",  # handle cluster imbalance
        random_state=100,
        n_jobs=-1 
    ))
])

# fit model
rf_med.fit(X_train, y_train)

# predictions
y_pred = rf_med.predict(X_test)

# evaluation
print("\nClassification Report (Cluster Prediction)")
print(classification_report(y_test, y_pred))

# prediction accuracy
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix")
print(cm)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[f"Cluster {i}" for i in range(k_chose)],
            yticklabels=[f"Cluster {i}" for i in range(k_chose)])
plt.xticks(ticks=[1, 2, 3, 4], labels=["1", "2", "3", "4"]) # same as barplots... I added to match cluster numbers
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (RF Clusters)")
plt.tight_layout()
plt.show()

# get feature importance
# get ohe feature names for categorical predictors
ohe = rf_med.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
encoded_cat_names = ohe.get_feature_names_out(categorical_cols)

# combine with numeric feature names
all_feature_names = list(encoded_cat_names) + numeric_cols

# get importances from RF
importances = rf_med.named_steps["rf"].feature_importances_

fi_rf = (
    pd.DataFrame({"feature": all_feature_names, "importance": importances})
      .sort_values("importance", ascending=False)
)

print("\nTop 20 Cluster Features:")
print(fi_rf.head(20))

