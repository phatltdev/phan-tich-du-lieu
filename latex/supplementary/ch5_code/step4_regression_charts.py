import pandas as pd, numpy as np, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)

with open("data/regression_preds.pkl","rb") as f:
    D = pickle.load(f)
y_test, dates_test, preds = D["y_test"], D["dates_test"], D["preds"]
importance = D["xgb_importance"]

colors = {"Linear Regression":"#2c7fb8","ARIMA(2,0,2)":"#e07b39","XGBoost":"#2ca25f"}

# 1. Line chart: Thuc te vs 3 mo hinh du bao (toan bo tap test)
fig, ax = plt.subplots(figsize=(13,4.5))
ax.plot(dates_test, y_test.values, label="Thực tế", color="black", linewidth=1.1)
for name,pred in preds.items():
    ax.plot(dates_test, pred, label=name, color=colors[name], linewidth=0.9, alpha=0.8)
ax.set_title("Dự báo Appliances (Wh/giờ): Thực tế vs 3 mô hình (tập test, 654 giờ)")
ax.legend()
plt.tight_layout(); plt.savefig("figs/fig5_forecast_full_test.png", dpi=140); plt.close()

# 2. Zoom 7 ngay dau tap test de de quan sat chi tiet
fig, ax = plt.subplots(figsize=(13,4.5))
mask = dates_test <= dates_test[0] + pd.Timedelta(days=7)
ax.plot(dates_test[mask], y_test.values[mask], label="Thực tế", color="black", linewidth=1.3, marker='o', markersize=2)
for name,pred in preds.items():
    ax.plot(dates_test[mask], np.array(pred)[mask], label=name, color=colors[name], linewidth=1.1)
ax.set_title("Dự báo Appliances - Zoom 7 ngày đầu tập test")
ax.legend()
plt.tight_layout(); plt.savefig("figs/fig6_forecast_zoom7days.png", dpi=140); plt.close()

# 3. Scatter Thuc te vs Du doan cho tung mo hinh
fig, axes = plt.subplots(1,3, figsize=(14,4.3))
for ax,(name,pred) in zip(axes, preds.items()):
    ax.scatter(y_test.values, pred, alpha=0.3, s=12, color=colors[name])
    lims = [0, max(y_test.max(), max(pred))]
    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Thực tế"); ax.set_ylabel("Dự đoán")
plt.tight_layout(); plt.savefig("figs/fig7_scatter_actual_vs_pred.png", dpi=140); plt.close()

# 4. Residual plot (XGBoost, mo hinh tot nhat)
fig, axes = plt.subplots(1,2, figsize=(11,4.3))
resid = y_test.values - preds["XGBoost"]
axes[0].scatter(dates_test, resid, alpha=0.4, s=10, color="#2ca25f")
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_title("Residual theo thời gian - XGBoost")
sns.histplot(resid, bins=30, kde=True, ax=axes[1], color="#2ca25f")
axes[1].set_title("Phân phối Residual - XGBoost")
plt.tight_layout(); plt.savefig("figs/fig8_residual_xgboost.png", dpi=140); plt.close()

# 5. Feature importance XGBoost
fig, ax = plt.subplots(figsize=(7,4.5))
imp_sorted = dict(sorted(importance.items(), key=lambda x:x[1]))
ax.barh(list(imp_sorted.keys()), list(imp_sorted.values()), color="#2ca25f")
ax.set_title("Mức độ quan trọng đặc trưng (XGBoost)")
plt.tight_layout(); plt.savefig("figs/fig9_feature_importance.png", dpi=140); plt.close()

print("DONE")
