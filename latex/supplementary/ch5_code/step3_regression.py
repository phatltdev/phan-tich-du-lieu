import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
sns.set_theme(style="whitegrid", font_scale=1.0)

df = pd.read_csv('/home/claude/ch3/energy/data/energy_clean_after_preprocessing.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# ---- Resample ve gio de bai toan du bao gon hon, on dinh hon (khop voi phan tich Ch3) ----
hourly = df.set_index('date').resample('h').mean(numeric_only=True)
hourly = hourly.dropna()
print("So diem du lieu theo gio:", len(hourly))

# ---- Feature engineering: dac trung thoi gian + lag ----
hourly['hour'] = hourly.index.hour
hourly['dayofweek'] = hourly.index.dayofweek
hourly['is_weekend'] = (hourly['dayofweek']>=5).astype(int)
hourly['lag_1'] = hourly['Appliances'].shift(1)
hourly['lag_24'] = hourly['Appliances'].shift(24)
hourly['rolling_mean_24'] = hourly['Appliances'].shift(1).rolling(24).mean()
hourly = hourly.dropna()

feature_cols = ['T_out','RH_out','Windspeed','hour','dayofweek','is_weekend','lag_1','lag_24','rolling_mean_24']
X = hourly[feature_cols]
y = hourly['Appliances']

# ---- Time Series Split: 80% dau de train, 20% cuoi de test (theo thu tu thoi gian) ----
split_idx = int(len(hourly)*0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = hourly.index[split_idx:]
print(f"Train: {len(X_train)} diem ({hourly.index[0]} - {hourly.index[split_idx-1]})")
print(f"Test:  {len(X_test)} diem ({hourly.index[split_idx]} - {hourly.index[-1]})")

RESULTS = []
preds = {}

# ============ 1. Linear Regression ============
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
lr = LinearRegression()
lr.fit(X_train_s, y_train)
pred_lr = lr.predict(X_test_s)
preds['Linear Regression'] = pred_lr

# ============ 2. ARIMA (chi dung chuoi Appliances, khong dung dac trung ngoai sinh) ============
arima_model = ARIMA(y_train.values, order=(2,0,2))
arima_fit = arima_model.fit()
pred_arima = arima_fit.forecast(steps=len(y_test))
preds['ARIMA(2,0,2)'] = pred_arima

# ============ 3. XGBoost ============
xgb = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)
preds['XGBoost'] = pred_xgb

# ---- Bang so sanh MAE / RMSE ----
for name, pred in preds.items():
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = np.mean(np.abs((y_test.values-pred)/np.maximum(y_test.values,1)))*100
    RESULTS.append({"Model":name,"MAE":mae,"RMSE":rmse,"MAPE_%":mape})
    print(f"{name}: MAE={mae:.3f} RMSE={rmse:.3f} MAPE={mape:.2f}%")

results_df = pd.DataFrame(RESULTS)
results_df.to_csv("data/table_regression_results.csv", index=False)

hourly.reset_index().to_csv("data/energy_hourly_features.csv", index=False)
import pickle
with open("data/regression_preds.pkl","wb") as f:
    pickle.dump({"y_test":y_test, "dates_test":dates_test, "preds":preds,
                 "xgb_importance": dict(zip(feature_cols, xgb.feature_importances_))}, f)
print("\nFeature importance (XGBoost):")
for k,v in sorted(zip(feature_cols, xgb.feature_importances_), key=lambda x:-x[1]):
    print(f"  {k}: {v:.4f}")
