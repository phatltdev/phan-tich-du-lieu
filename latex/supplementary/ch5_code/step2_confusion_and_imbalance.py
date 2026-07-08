import pandas as pd, numpy as np, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

sns.set_theme(style="whitegrid", font_scale=1.0)

with open("data/confusion_matrices.pkl","rb") as f:
    D = pickle.load(f)

def plot_cms(key, title_prefix, fname):
    cms, labels = D[key]
    fig, axes = plt.subplots(1,3, figsize=(14,4.3))
    for ax,(mname,cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels, cbar=False)
        ax.set_title(mname, fontsize=10)
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
    fig.suptitle(f"Ma trận nhầm lẫn - {title_prefix}")
    plt.tight_layout()
    plt.savefig(f"figs/{fname}", dpi=140)
    plt.close()

plot_cms("bc", "Breast Cancer (Diagnosis)", "fig1_cm_breastcancer.png")
plot_cms("heart", "Heart Disease (Target)", "fig2_cm_heart.png")
plot_cms("diab", "CDC Diabetes (3-class)", "fig3_cm_diabetes.png")

# ---- Phan tich rieng cho Diabetes: per-class recall tu Random Forest de minh hoa mat can bang ----
cms_diab, labels_diab = D["diab"]
cm_rf = cms_diab["Random Forest"]
per_class_recall = cm_rf.diagonal()/cm_rf.sum(axis=1)
per_class_df = pd.DataFrame({"Lop":labels_diab, "So_mau_test":cm_rf.sum(axis=1), "Recall":per_class_recall.round(4)})
per_class_df.to_csv("data/table_diabetes_perclass_recall.csv", index=False)
print(per_class_df.to_string(index=False))

# bar chart per-class recall
fig, ax = plt.subplots(figsize=(6,4.5))
sns.barplot(data=per_class_df, x="Lop", y="Recall", color="#2c7fb8", ax=ax)
ax.set_title("Recall theo từng lớp - CDC Diabetes (Random Forest)\n(minh họa ảnh hưởng mất cân bằng lớp)")
ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig("figs/fig4_perclass_recall_diabetes.png", dpi=140)
plt.close()
print("DONE")
