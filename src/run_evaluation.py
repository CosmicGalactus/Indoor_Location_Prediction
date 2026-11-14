import os
import sys
sys.path.insert(0, os.path.abspath('.'))
os.environ['MPLCONFIGDIR'] = '/tmp/mpl'

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from src.hierarchical_knn import HierarchicalKNNPositioner

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 1. Load Data
data_path = 'Datasets/processed/preprocessed_wifi_data.joblib'
data = joblib.load(data_path)

active_waps = data['active_waps']
n_waps = len(active_waps)

X_train_waps = data['X_train'][:, :n_waps]
X_val_waps = data['X_val'][:, :n_waps]

y_train_bldg = data['y_train_bldg']
y_val_bldg = data['y_val_bldg']
y_train_floor = data['y_train_floor']
y_val_floor = data['y_val_floor']
y_train_coords = data['y_train_coords']
y_val_coords = data['y_val_coords']

print("--- Running Distance Metric Ablation ---")
def eval_flat_knn(xtr, xte, metric, k=5, weights='distance'):
    clf_b = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights, n_jobs=-1).fit(xtr, y_train_bldg)
    pb = clf_b.predict(xte)
    
    clf_f = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights, n_jobs=-1).fit(xtr, y_train_floor)
    pf = clf_f.predict(xte)
    
    reg_c = KNeighborsRegressor(n_neighbors=k, metric=metric, weights=weights, n_jobs=-1).fit(xtr, y_train_coords)
    pc = reg_c.predict(xte)
    
    b_acc = accuracy_score(y_val_bldg, pb) * 100
    f_acc = accuracy_score(y_val_floor, pf) * 100
    both = np.mean((pb == y_val_bldg) & (pf == y_val_floor)) * 100
    errs = np.sqrt(np.sum((y_val_coords - pc)**2, axis=1))
    
    return {
        'Building Acc (%)': b_acc,
        'Floor Acc (%)': f_acc,
        'Both Match (%)': both,
        'Mean Error (m)': np.mean(errs),
        'Median Error (m)': np.median(errs),
        '75th %ile Error (m)': np.percentile(errs, 75),
        '90th %ile Error (m)': np.percentile(errs, 90)
    }

benchmarks = []
# 1. Baseline with count features
res_base = eval_flat_knn(data['X_train'], data['X_val'], 'euclidean', k=5, weights='uniform')
res_base['Model'] = 'Baseline KNN (k=5, Euclidean, with Count features)'
benchmarks.append(res_base)

# 2. Pure WAP Euclidean
res_euc = eval_flat_knn(X_train_waps, X_val_waps, 'euclidean', k=5)
res_euc['Model'] = 'Euclidean KNN (WAP Only)'
benchmarks.append(res_euc)

# 3. Manhattan
res_man = eval_flat_knn(X_train_waps, X_val_waps, 'cityblock', k=5)
res_man['Model'] = 'Manhattan KNN (WAP Only)'
benchmarks.append(res_man)

# 4. Cosine
res_cos = eval_flat_knn(X_train_waps, X_val_waps, 'cosine', k=5)
res_cos['Model'] = 'Cosine KNN (WAP Only, k=5)'
benchmarks.append(res_cos)

# 5. Bray-Curtis
res_bc = eval_flat_knn(X_train_waps, X_val_waps, 'braycurtis', k=5)
res_bc['Model'] = 'Bray-Curtis KNN (WAP Only, k=5)'
benchmarks.append(res_bc)

# 6. Champion Hierarchical KNN (Cosine + Bray-Curtis with gamma=2.0)
hknn = HierarchicalKNNPositioner(
    gamma=2.0,
    k_bldg=5,
    k_floor=9,
    k_coords=7,
    metric_bldg='cosine',
    metric_floor='cosine',
    metric_coords='braycurtis',
    weights='distance'
)
hknn.fit(X_train_waps, y_train_bldg, y_train_floor, y_train_coords)
pred_bldg, pred_floor, pred_coords = hknn.predict(X_val_waps)

final_bldg_acc = accuracy_score(y_val_bldg, pred_bldg) * 100
final_floor_acc = accuracy_score(y_val_floor, pred_floor) * 100
final_both_acc = np.mean((pred_bldg == y_val_bldg) & (pred_floor == y_val_floor)) * 100
hknn_errors = np.sqrt(np.sum((y_val_coords - pred_coords)**2, axis=1))

res_hknn = {
    'Model': 'Hierarchical KNN Champion (Cosine + Bray-Curtis, γ=2.0)',
    'Building Acc (%)': final_bldg_acc,
    'Floor Acc (%)': final_floor_acc,
    'Both Match (%)': final_both_acc,
    'Mean Error (m)': np.mean(hknn_errors),
    'Median Error (m)': np.median(hknn_errors),
    '75th %ile Error (m)': np.percentile(hknn_errors, 75),
    '90th %ile Error (m)': np.percentile(hknn_errors, 90)
}
benchmarks.append(res_hknn)

df_bench = pd.DataFrame(benchmarks)
print(df_bench.to_string(index=False))

# Visualizations
os.makedirs('figures', exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Cumulative Error Distribution (ECDF)
sorted_hknn_errors = np.sort(hknn_errors)
p = 1.0 * np.arange(len(sorted_hknn_errors)) / (len(sorted_hknn_errors) - 1)

axes[0].plot(sorted_hknn_errors, p * 100, label='Hierarchical KNN (Champion)', color='#2ca02c', lw=2.5)
axes[0].axvline(np.median(hknn_errors), color='#2ca02c', linestyle='--', alpha=0.7, label=f'Median: {np.median(hknn_errors):.2f}m')
axes[0].axvline(np.percentile(hknn_errors, 90), color='crimson', linestyle=':', alpha=0.7, label=f'90th %ile: {np.percentile(hknn_errors, 90):.2f}m')
axes[0].set_xlim(0, 40)
axes[0].set_xlabel('Euclidean Positioning Error (meters)', fontsize=12)
axes[0].set_ylabel('Cumulative Percentage (%)', fontsize=12)
axes[0].set_title('Empirical Cumulative Error Distribution (ECDF)', fontsize=14, fontweight='bold')
axes[0].legend(loc='lower right', frameon=True)
axes[0].grid(True, alpha=0.4)

# Plot 2: Floor Confusion Matrix
cm = confusion_matrix(y_val_floor, pred_floor)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1],
            xticklabels=[f'Floor {i}' for i in range(5)],
            yticklabels=[f'Floor {i}' for i in range(5)])
axes[1].set_title(f'Champion Floor Confusion Matrix (Floor Acc: {final_floor_acc:.2f}%)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted Floor', fontsize=12)
axes[1].set_ylabel('Ground Truth Floor', fontsize=12)

plt.tight_layout()
fig_path = 'figures/advanced_knn_performance.png'
plt.savefig(fig_path, dpi=300)
plt.close()
print(f'Saved visualization to {fig_path}')

# Save champion model asset
os.makedirs('Reports', exist_ok=True)
model_path = 'Reports/hierarchical_knn_champion.joblib'
joblib.dump({
    'model': hknn,
    'metrics': res_hknn,
    'active_waps': active_waps,
    'errors': hknn_errors
}, model_path)
print(f'Saved champion model to {model_path}')
