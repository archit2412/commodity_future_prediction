import os
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

print("--- Loading and preparing data... ---")

try:
    train_df = pd.read_csv('train.csv')
    train_labels_df = pd.read_csv('train_labels.csv')
    target_pairs_df = pd.read_csv('target_pairs.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure your CSV files are in the same directory.")
    exit()

# Melt the labels DataFrame to a long format
train_labels_melted_df = train_labels_df.melt(
    id_vars=['date_id'],
    var_name='target',
    value_name='target_value'
)

# Merge all data
training_data = pd.merge(train_labels_melted_df, target_pairs_df, on='target', how='left')
training_data = pd.merge(training_data, train_df, on='date_id', how='left')
feature_cols = [col for col in train_df.columns if col != 'date_id']

# --- STEP 2: Time-Based Data Split ---
print("\n--- Performing a time-based data split... ---")
training_data = training_data.sort_values(by='date_id')
unique_dates = training_data['date_id'].unique()
if len(unique_dates) < 2:
    print("Not enough unique dates for time-based split.")
    exit()

split_point = int(len(unique_dates) * 0.8)
split_date_id = unique_dates[split_point]

train_set = training_data[training_data['date_id'] < split_date_id].copy()
val_set = training_data[training_data['date_id'] >= split_date_id].copy()

print(f"Training data size: {len(train_set)} rows")
print(f"Validation data size: {len(val_set)} rows")

# --- STEP 3: Train and Tune Models on the Training Set (Optuna, minimize MAE) ---
print("\n--- Starting Bayesian Hyperparameter Tuning and Training (Optuna, MAE) ... ---")
models = {}
results = []
target_list = train_set['target'].unique()

for i, target_name in enumerate(target_list):
    print(f"Tuning and training model {i+1}/{len(target_list)} for: {target_name}")
   
    target_train_data = train_set[train_set['target'] == target_name].dropna(subset=['target_value'])
    if target_train_data.empty:
        print(f"  ⚠️ No training data for {target_name}, skipping.")
        continue
   
    X_train = target_train_data[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_train = target_train_data['target_value'].astype(float)

    def objective(trial):
        params = {
            'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1]),
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300]),
            'num_leaves': trial.suggest_categorical('num_leaves', [15, 31, 63]),
            'max_depth': trial.suggest_categorical('max_depth', [-1, 5, 10]),
            'min_child_samples': trial.suggest_categorical('min_child_samples', [10, 20, 50]),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        model = LGBMRegressor(**params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        return -scores.mean()

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=20, n_jobs=-1, show_progress_bar=False)

    best_params = study.best_params
    final_params = best_params.copy()
    final_params.update({'random_state': 42, 'verbose': -1, 'n_jobs': -1})
    model = LGBMRegressor(**final_params)
    model.fit(X_train, y_train)

    # compute train MAE for overfit/underfit detection
    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)

    models[target_name] = model
    results.append({
        'target': target_name,
        'best_params': best_params,
        'train_MAE': train_mae
    })
    print(f"  Trained {target_name} | train_MAE: {train_mae:.6f}")

# --- STEP 4: Evaluate on the Validation Set ---
print("\n--- Evaluating tuned models on the validation set... ---")
all_preds = []
all_true = []
final_results = []

for r in results:
    target_name = r['target']
    model = models.get(target_name)
    if model is None:
        continue

    target_val_data = val_set[val_set['target'] == target_name].dropna(subset=['target_value'])
    if target_val_data.empty:
        print(f"  ⚠️ No validation data for {target_name}, skipping evaluation.")
        continue

    X_val = target_val_data[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_val = target_val_data['target_value'].astype(float).values
    y_val_pred = model.predict(X_val)

    val_mae = mean_absolute_error(y_val, y_val_pred)

    all_preds.extend(y_val_pred.tolist())
    all_true.extend(y_val.tolist())

    # determine overfit/underfit: compare train_MAE vs val_MAE
    train_mae = r.get('train_MAE', np.nan)
    status = 'Unknown'
    if not np.isnan(train_mae):
        # if train MAE much lower than val -> overfit; if train much higher -> underfit
        if train_mae < 0.7 * val_mae:
            status = 'Overfitting'
        elif train_mae > 1.3 * val_mae:
            status = 'Underfitting'
        else:
            status = 'Good Fit'

    final_results.append({
        'target': target_name,
        'best_params': r['best_params'],
        'train_MAE': train_mae,
        'val_MAE': val_mae,
        'status': status,
        'n_val': len(y_val)
    })

    print(f"  {target_name} | train_MAE: {train_mae:.6f} | val_MAE: {val_mae:.6f} | status: {status}")

# --- STEP 5: Aggregate metrics (MAE) and Sharpe ratios ---
if len(all_true) == 0:
    print("No validation predictions collected; cannot compute aggregate MAE / Sharpe.")
else:
    y_true = np.array(all_true, dtype=float)      
    y_pred = np.array(all_preds, dtype=float)

    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\nFinal Mean Absolute Error (MAE) on Validation Set: {mae:.8f}")

    # Sharpe computations on prediction errors (alpha = pred - true)
    alpha = y_pred - y_true
    eps = 1e-12

    std_alpha = np.std(alpha, ddof=0)
    if std_alpha < eps:
        print("Signed Sharpe: std(alpha) is zero; cannot compute Sharpe.")
    else:
        sharpe_signed = (alpha.mean() / std_alpha) * np.sqrt(252)
        print(f"Signed Sharpe (annualized): {sharpe_signed:.6f}")

    neg_abs = -np.abs(alpha)
    std_neg_abs = np.std(neg_abs, ddof=0)
    if std_neg_abs < eps:
        print("Neg-Abs Sharpe: std is zero; cannot compute Sharpe.")
    else:
        sharpe_neg_abs = (neg_abs.mean() / std_neg_abs) * np.sqrt(252)
        print(f"Neg-Abs Sharpe (annualized): {sharpe_neg_abs:.6f}")

# --- STEP 6: Save results ---
final_df = pd.DataFrame(final_results)
if not final_df.empty:
    out_path = os.path.join(os.path.dirname(__file__), 'mae_sharpe_results.csv')
    final_df.to_csv(out_path, index=False)
    print(f"\nPer-target results saved to: {out_path}")
