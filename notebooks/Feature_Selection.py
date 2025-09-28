import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, chi2, SelectKBest 
from sklearn.model_selection import train_test_split 
import Data_PreProcessing as data_prep 

x = data_prep.df_scaled.drop('target', axis=1)
y = data_prep.df_scaled['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# 1 - Use Feature Importance (Random Forest) to rank variables
feature_importances = pd.Series(rf_model.feature_importances_, index=x_train.columns)
ranked_rf = feature_importances.sort_values(ascending=False)

# 2 - Apply Recursive Feature Elimination (RFE) to select the best predictors.
n_rfe_features = 10 
rfe_selector = RFE(estimator=rf_model, n_features_to_select=n_rfe_features, step=1)
rfe_selector.fit(x_train, y_train)

rfe_selected_features = x_train.columns[rfe_selector.support_].tolist()

rfe_ranking = pd.Series(rfe_selector.ranking_, index=x_train.columns).sort_values()

# 3 - Use Chi-Square Test to check feature significance.
n_chi2_features = 15
chi2_selector = SelectKBest(score_func=chi2, k=n_chi2_features)
chi2_selector.fit(x_train, y_train)

chi2_scores = pd.Series(chi2_selector.scores_, index=x_train.columns)
chi2_pvalues = pd.Series(chi2_selector.pvalues_, index=x_train.columns)

significant_chi2_features = chi2_pvalues[chi2_pvalues < 0.05].index.tolist()

# 4 - Select only the most relevant features for modeling.
n_final_features = 15 
final_features = ranked_rf.head(n_final_features).index.tolist()

x_final = x[final_features]

if __name__ == '__main__':
    print(f'Feature Selection Summary')
    print(f'Original number of features: {x.shape[1]}')
    print(f'Final selected number of features: {len(final_features)}')
    
    print(f'Random Forest Feature Importance (Top {n_final_features}):')
    print(ranked_rf.head(n_final_features))
    
    print(f'RFE Selected Features (Top {n_rfe_features}):')
    print(rfe_selected_features)

    print(f'Chi-Square Significant Features (p < 0.05):')
    print(significant_chi2_features)
    
    print(f'Final Feature Set For Modeling')
    print(final_features)
    
    # Visualize Feature Importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x=ranked_rf.head(n_final_features).values, 
                y=ranked_rf.head(n_final_features).index, 
                palette="viridis")
    plt.title(f'Top {n_final_features} Features by Random Forest Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.show()