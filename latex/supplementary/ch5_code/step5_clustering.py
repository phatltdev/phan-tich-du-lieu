import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

np.random.seed(42)
sns.set_theme(style="whitegrid", font_scale=1.0)
SUMMARY = []

def elbow_silhouette(X_scaled, name, fname, k_range=range(2,9)):
    inertias, sils = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X_scaled, km.labels_))
    fig, axes = plt.subplots(1,2, figsize=(11,4.3))
    axes[0].plot(list(k_range), inertias, marker='o', color="#2c7fb8")
    axes[0].set_title(f"Elbow Method - {name}")
    axes[0].set_xlabel("Số cụm K"); axes[0].set_ylabel("Inertia")
    axes[1].plot(list(k_range), sils, marker='o', color="#d62728")
    axes[1].set_title(f"Silhouette Score - {name}")
    axes[1].set_xlabel("Số cụm K"); axes[1].set_ylabel("Silhouette Score")
    plt.tight_layout(); plt.savefig(f"figs/{fname}", dpi=140); plt.close()
    best_k = list(k_range)[np.argmax(sils)]
    return best_k, dict(zip(k_range,inertias)), dict(zip(k_range,sils))

def compare_algorithms(X_scaled, k, name):
    results = {}
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    results["KMeans"] = (km.labels_, silhouette_score(X_scaled, km.labels_))
    agg = AgglomerativeClustering(n_clusters=k).fit(X_scaled)
    results["Agglomerative"] = (agg.labels_, silhouette_score(X_scaled, agg.labels_))
    gmm = GaussianMixture(n_components=k, random_state=42).fit(X_scaled)
    gmm_labels = gmm.predict(X_scaled)
    results["Gaussian Mixture"] = (gmm_labels, silhouette_score(X_scaled, gmm_labels))
    for algo,(labels,sil) in results.items():
        SUMMARY.append({"Dataset":name,"Algorithm":algo,"K":k,"Silhouette":sil})
        print(f"{name} - {algo}: Silhouette={sil:.4f}")
    return results

def pca_plot(X_scaled, labels, name, fname, true_label=None, true_label_name=""):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    n_plots = 2 if true_label is not None else 1
    fig, axes = plt.subplots(1,n_plots, figsize=(6*n_plots,5))
    if n_plots==1: axes=[axes]
    sc = axes[0].scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="tab10", alpha=0.6, s=15)
    axes[0].set_title(f"PCA 2D - Cụm KMeans ({name})")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    if true_label is not None:
        codes = pd.Categorical(true_label).codes
        axes[1].scatter(X_pca[:,0], X_pca[:,1], c=codes, cmap="Set1", alpha=0.6, s=15)
        axes[1].set_title(f"PCA 2D - Nhãn thật ({true_label_name})")
        axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    plt.tight_layout(); plt.savefig(f"figs/{fname}", dpi=140); plt.close()

# ================= DATA A: BREAST CANCER =================
print("\n===== DATA A: Breast Cancer =====")
bc = pd.read_csv('/home/claude/ch3/data/bc_clean_after_preprocessing.csv')
feat_bc = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean"]
X_bc = StandardScaler().fit_transform(bc[feat_bc])
k_bc, inertia_bc, sil_bc = elbow_silhouette(X_bc, "Breast Cancer (Data A)", "fig10_elbow_sil_bc.png")
print("K toi uu (Silhouette):", k_bc)
res_bc = compare_algorithms(X_bc, k_bc, "Breast Cancer")
labels_bc = res_bc["KMeans"][0]
pca_plot(X_bc, labels_bc, "Breast Cancer", "fig11_pca_bc.png", bc["diagnosis"], "diagnosis")
bc["cluster"] = labels_bc
profile_bc = bc.groupby("cluster")[feat_bc].mean().round(2)
profile_bc["n"] = bc.groupby("cluster").size()
profile_bc["ty_le_malignant_%"] = (bc.groupby("cluster")["diagnosis"].apply(lambda x:(x=="Malignant (Ác tính)").mean()*100)).round(2)
profile_bc.to_csv("data/table_cluster_profile_bc.csv")
print(profile_bc)

# ================= DATA B: APPLIANCES ENERGY =================
print("\n===== DATA B: Appliances Energy =====")
energy = pd.read_csv('/home/claude/ch4/data/energy_hourly_features.csv')
feat_energy = ["Appliances","T_out","RH_out","Windspeed","hour","dayofweek"]
X_energy = StandardScaler().fit_transform(energy[feat_energy])
k_en, inertia_en, sil_en = elbow_silhouette(X_energy, "Appliances Energy (Data B)", "fig12_elbow_sil_energy.png")
print("K toi uu (Silhouette):", k_en)
res_en = compare_algorithms(X_energy, k_en, "Appliances Energy")
labels_en = res_en["KMeans"][0]
pca_plot(X_energy, labels_en, "Appliances Energy", "fig13_pca_energy.png")
energy["cluster"] = labels_en
profile_en = energy.groupby("cluster")[feat_energy].mean().round(2)
profile_en["n"] = energy.groupby("cluster").size()
profile_en.to_csv("data/table_cluster_profile_energy.csv")
print(profile_en)

# ================= DATA C: HAM10000 =================
print("\n===== DATA C: HAM10000 =====")
ham = pd.read_csv('/home/claude/ch3/ham10000/data/ham10000_features.csv')
feat_ham = ["age","mean_brightness","std_brightness","mean_R","mean_G","mean_B","lesion_area_ratio"]
ham_valid = ham.dropna(subset=feat_ham).reset_index(drop=True)
X_ham = StandardScaler().fit_transform(ham_valid[feat_ham])
k_ham, inertia_ham, sil_ham = elbow_silhouette(X_ham, "HAM10000 (Data C)", "fig14_elbow_sil_ham.png")
print("K toi uu (Silhouette):", k_ham)
res_ham = compare_algorithms(X_ham, k_ham, "HAM10000")
labels_ham = res_ham["KMeans"][0]
pca_plot(X_ham, labels_ham, "HAM10000", "fig15_pca_ham.png", ham_valid["group"], "group")
ham_valid["cluster"] = labels_ham
profile_ham = ham_valid.groupby("cluster")[feat_ham].mean().round(2)
profile_ham["n"] = ham_valid.groupby("cluster").size()
profile_ham["ty_le_ac_tinh_%"] = (ham_valid.groupby("cluster")["group"].apply(lambda x:(x=="Ác tính/Tiền ung thư").mean()*100)).round(2)
profile_ham.to_csv("data/table_cluster_profile_ham.csv")
print(profile_ham)

summary_df = pd.DataFrame(SUMMARY)
summary_df.to_csv("data/table_clustering_algo_comparison.csv", index=False)
print("\n\nBang tong hop:\n", summary_df.to_string(index=False))

optimal_k_df = pd.DataFrame({"Dataset":["Breast Cancer","Appliances Energy","HAM10000"],
                              "K_toi_uu":[k_bc,k_en,k_ham]})
optimal_k_df.to_csv("data/table_optimal_k.csv", index=False)
