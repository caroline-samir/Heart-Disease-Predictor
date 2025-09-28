import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc
from scipy.stats import randint
import matplotlib.pyplot as plt
import joblib
import Feature_Selection as fs 

x = fs.x_final
y = fs.y

# Baseline F1-scores obtained from Supervised_Learning.py for comparison
baseline_metrics = {
    'Random Forest (Baseline)': {'F1-score': 0.9065, 'Accuracy': 0.9024, 'AUC-score': 0.9500},
    'Decision Tree (Baseline)': {'F1-score': 0.8869, 'Accuracy': 0.8780, 'AUC-score': 0.9480}
}
final_comparison_metrics = baseline_metrics.copy()

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# function to evaluate and store metrics
def evaluate_model_performance(name, model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    
    # Core Metrics
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    # AUC Score
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = auc(fpr, tpr)
    
    return {'F1-score': f1, 'Accuracy': acc, 'AUC-score': auc_score}


# 1.1 - Use GridSearchCV to optimize model (Random Forest) hyperparameters.

rf_model = RandomForestClassifier(random_state=42)

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7, 9],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [3, 5]
}

grid_search_rf = GridSearchCV(
    estimator=rf_model, 
    param_grid=rf_param_grid, 
    scoring='f1', 
    cv=3, 
    verbose=1, 
    n_jobs=-1
)

grid_search_rf.fit(x_train, y_train)

best_rf_params = grid_search_rf.best_params_
best_rf_score = grid_search_rf.best_score_
best_rf_model = grid_search_rf.best_estimator_

# 1.2 - Use RandomizedSearchCV to optimize model (Decision Tree) hyperparameters.
dt_model = DecisionTreeClassifier(random_state=42)

dt_param_dist = {
    'max_depth': randint(3, 15),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 15),
    'criterion': ['gini', 'entropy']
}

random_search_dt = RandomizedSearchCV(
    estimator=dt_model, 
    param_distributions=dt_param_dist, 
    n_iter=50,
    scoring='f1', 
    cv=3, 
    verbose=1, 
    random_state=42, 
    n_jobs=-1
)

random_search_dt.fit(x_train, y_train)

best_dt_params = random_search_dt.best_params_
best_dt_score = random_search_dt.best_score_
best_dt_model = random_search_dt.best_estimator_


# ----------------------------------------------------
# --- STEP 2: Compare Optimized Models with Baseline ---
# ----------------------------------------------------
if __name__ == '__main__':
    # printing outputs
    print(f"Starting Hyperparameter Tuning on {x_train.shape[0]} training samples.") #printing the initials

    print(f'\n1.1 Starting GridSearchCV for Random Forest\n')
    # printing GridSearchCV results
    print(f'\nRandom Forest Best CV F1 Score: {best_rf_score:.4f}')
    print(f'Random Forest Best Parameters: {best_rf_params}')

    print(f'\n1.2 Starting RandomizedSearchCV for Decision Tree\n')
    # printing RandomizedSearchCV results
    print(f'\nDecision Tree Best CV F1 Score: {best_dt_score:.4f}')
    print(f'Decision Tree Best Parameters: {best_dt_params}')

    # 2 - Compare optimized models with baseline performance.
    # 2.1 - Evaluate Optimized Random Forest on Test Set
    rf_optimized_metrics = evaluate_model_performance(
        'Random Forest (Optimized)', 
        best_rf_model, 
        x_test, 
        y_test
    )
    final_comparison_metrics['Random Forest (Optimized)'] = rf_optimized_metrics

    # 2.2 - Evaluate Optimized Decision Tree on Test Set
    dt_optimized_metrics = evaluate_model_performance(
        'Decision Tree (Optimized)', 
        best_dt_model, 
        x_test, 
        y_test
    )
    final_comparison_metrics['Decision Tree (Optimized)'] = dt_optimized_metrics

    # 2.3 - Create and Print Final Comparison Table
    comparison_df = pd.DataFrame(final_comparison_metrics).T

    print(f'\nFinal Performance Comparison: Baseline vs. Optimized Models\n')
    
    # Sort to see the best overall performance
    comparison_df_sorted = comparison_df.sort_values(by='F1-score', 
                                                     ascending=False)
    
    print(comparison_df_sorted.to_markdown(floatfmt=".4f"))


    # Model Export & Deployment
    # We choose the best performing model, the Optimized Random Forest, for persistence.
    print(f'\n3. Saving the Final Optimized Random Forest Model\n')
    
    # 3.1 - Save the trained model using joblib (.pkl format).
    model_filepath = 'final_model.pkl'
    try:
        joblib.dump(best_rf_model, model_filepath)
        print(f'\nOptimized Random Forest Model saved successfully to: {model_filepath}\n')
    except Exception as e:
        print(f'\nError saving model: {e}\n')

    # 3.2 - Ensure reproducibility by saving model pipeline (preprocessing + model).
    feature_names = list(x_train.columns)
    features_filepath = 'optimized_random_forest_features.pkl'
    try:
        joblib.dump(feature_names, features_filepath)
        print(f'\nFeature list (pipeline state) saved successfully to: {features_filepath}\n')
    except Exception as e:
        print(f'\nError saving feature list: {e}\n')