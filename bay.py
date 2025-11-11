import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# ...existing code...
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
split_point = int(len(training_data['date_id'].unique()) * 0.8)
split_date_id = training_data['date_id'].unique()[split_point]

train_set = training_data[training_data['date_id'] < split_date_id]
val_set = training_data[training_data['date_id'] >= split_date_id]

print(f"Training data size: {len(train_set)} rows")
print(f"Validation data size: {len(val_set)} rows")

# --- STEP 3: Define Hyperparameter Search Space ---
# The search space values are used by Optuna suggestions below.

# --- STEP 4: Train and Tune Models on the Training Set (Bayesian Optimization with Optuna) ---
print("\n--- Starting Bayesian Hyperparameter Tuning and Training (Optuna)... ---")
models = {}
target_list = train_set['target'].unique()
all_mse = [] # List to track MSE for each target

for i, target_name in enumerate(target_list):
    print(f"Tuning and training model {i+1}/{len(target_list)} for: {target_name}")
   
    target_train_data = train_set[train_set['target'] == target_name].dropna(subset=['target_value'])
    if target_train_data.empty:
        continue
   
    X_train = target_train_data[feature_cols]
    y_train = target_train_data['target_value']

    # Ensure numeric features
    X_train = X_train.apply(pd.to_numeric, errors='coerce')

    # Optuna objective: minimize CV MSE
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1]),
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300]),
            'num_leaves': trial.suggest_categorical('num_leaves', [15, 31, 63]),
            'max_depth': trial.suggest_categorical('max_depth', [-1, 5, 10]),
            'min_child_samples': trial.suggest_categorical('min_child_samples', [10, 20, 50]),
            'random_state': 42,
            'verbose': -1,
            # keep n_jobs small during CV to avoid nested parallelism issues
            'n_jobs': -1
        }
        model = LGBMRegressor(**params)
        # use 3-fold CV like before; return mean MSE
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        return -scores.mean()

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=30, n_jobs=-1, show_progress_bar=False)

    best_params = study.best_params
    # instantiate final model with more parallelism for final fit
    final_params = best_params.copy()
    final_params.update({'random_state': 42, 'verbose': -1, 'n_jobs': -1})
    best_model = LGBMRegressor(**final_params)
    best_model.fit(X_train, y_train)

    models[target_name] = best_model

# --- STEP 5: Evaluate on the Validation Set ---
print("\n--- Evaluating tuned models on the validation set... ---")
all_preds = []
all_true = []

for target_name, model in models.items():
    target_val_data = val_set[val_set['target'] == target_name].dropna(subset=['target_value'])
    if target_val_data.empty: continue
   
    X_val = target_val_data[feature_cols]
    y_val_true = target_val_data['target_value']
   
    # Ensure validation features are also numeric
    X_val = X_val.apply(pd.to_numeric, errors='coerce')
   
    y_val_pred = model.predict(X_val)

    all_preds.extend(y_val_pred)
    all_true.extend(y_val_true)

# Calculate and print the overall Mean Squared Error (MSE)
mse = mean_squared_error(all_true, all_preds)
print(f"\nFinal Mean Squared Error (MSE) on Validation Set after Bayesian Tuning: {mse:.8f}")
# ...existing code...
