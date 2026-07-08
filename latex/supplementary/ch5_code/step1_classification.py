import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, classification_report, roc_auc_score)

np.random.seed(42)
sns.set_theme(style="whitegrid", font_scale=1.0)
RESULTS = []

def run_classification(name, X, y, labels, avg="binary", subsample_n=None):
    print(f"\n===== {name} =====")
    if subsample_n and len(X) > subsample_n:
        idx = X.sample(subsample_n, random_state=42, weights=None).index
        # stratified subsample
        from sklearn.model_selection import train_test_split as tts
        X_sub, _, y_sub, _ = tts(X, y, train_size=subsample_n, stratify=y, random_state=42)
        X, y = X_sub.reset_index(drop=True), y_sub.reset_index(drop=True)
        print(f"(Da lay mau con {subsample_n} dong do kich thuoc lon)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "SVM (RBF)": SVC(kernel="rbf", random_state=42, class_weight="balanced")
    }

    cms = {}
    for mname, model in models.items():
        if mname == "Random Forest":
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        cms[mname] = cm

        RESULTS.append({"Dataset":name,"Model":mname,"Accuracy":acc,"Precision":prec,"Recall":rec,"F1":f1,
                         "n_train":len(X_train),"n_test":len(X_test)})
        print(f"{mname}: Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")

    return cms, labels, y_test

# ============ 1. BREAST CANCER (Group A - binary) ============
bc = pd.read_csv('/home/claude/ch3/data/bc_clean_after_preprocessing.csv')
X_bc = bc.drop(columns=["diagnosis"])
y_bc = (bc["diagnosis"]=="Malignant (Ác tính)").astype(int)  # 1=Malignant
cms_bc, labels_bc, ytest_bc = run_classification("Breast Cancer (Diagnosis)", X_bc, y_bc,
                                                   ["Benign","Malignant"], avg="binary")

# ============ 2. HEART DISEASE (Group A - binary) ============
heart = pd.read_csv('/home/claude/ch3/heart/data/heart_clean_after_preprocessing.csv')
feat_cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
heart_feat = heart[feat_cols].copy()
heart_feat["thal"] = heart_feat["thal"].fillna(heart_feat["thal"].mode()[0])
y_heart = heart["target"]
cms_heart, labels_heart, ytest_heart = run_classification("Heart Disease (Target)", heart_feat, y_heart,
                                                            ["No Disease","Disease"], avg="binary")

# ============ 3. CDC DIABETES (Group A - 3-class, imbalanced, subsample) ============
diab = pd.read_csv('/home/claude/ch3/diabetes/data/diabetes_clean_after_preprocessing.csv')
feat_cols_d = [c for c in diab.columns if c not in ["Diabetes_012","diab_label"]]
X_diab = diab[feat_cols_d]
y_diab = diab["Diabetes_012"].astype(int)
cms_diab, labels_diab, ytest_diab = run_classification("CDC Diabetes (3-class)", X_diab, y_diab,
                                                         ["No Diabetes","Prediabetes","Diabetes"],
                                                         avg="macro", subsample_n=30000)

results_df = pd.DataFrame(RESULTS)
results_df.to_csv("data/table_classification_results.csv", index=False)
print("\n\n", results_df.to_string(index=False))

import pickle
with open("data/confusion_matrices.pkl","wb") as f:
    pickle.dump({"bc":(cms_bc,labels_bc),"heart":(cms_heart,labels_heart),"diab":(cms_diab,labels_diab)}, f)
