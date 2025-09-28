import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, classification_report
import Feature_Selection as fs 

x = fs.x_final
y = fs.y

final_metrics = {}
roc_data = {}

# 1 - Split the dataset into training (80%) and testing (20%) sets.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Create a function to store and evaluate data
def evaluate_and_store(name, model, y_true, y_pred, y_proba=None):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-score': f1_score(y_true, y_pred),
        'AUC-score': 0.0
    }
    
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        metrics['AUC-score'] = auc(fpr, tpr)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr}
    
    final_metrics[name] = metrics
    
    print(f'\n{name} Results')
    print(f'Accuracy: {metrics['Accuracy']:.4f}')
    print(f'F1-score: {metrics['F1-score']:.4f}')
    print(f'AUC-score: {metrics['AUC-score']:.4f}')
    print(classification_report(y_true, y_pred))

# 2 & 3 - Train and Evaluate Models
# a . Logistic Regression
model_name1 = "Logistic Regression"
lr_model = LogisticRegression(random_state=42, solver='liblinear')
lr_model.fit(x_train, y_train)
y_pred_lr = lr_model.predict(x_test)
y_proba_lr = lr_model.predict_proba(x_test)[:, 1]

# b . Decision Tree
model_name2 = "Decision Tree"
dt_model = DecisionTreeClassifier(max_depth = 5 ,
                                  min_samples_leaf = 10,
                                  random_state=42)
dt_model.fit(x_train, y_train)
y_pred_dt = dt_model.predict(x_test)
y_proba_dt = dt_model.predict_proba(x_test)[:, 1]

# c . Random Forest
model_name3 = "Random Forest"
rf_model = RandomForestClassifier(n_estimators=100, 
                                  max_depth=5,
                                  min_samples_split=10,
                                  min_samples_leaf=5,
                                  random_state=42)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)
y_proba_rf = rf_model.predict_proba(x_test)[:, 1]

# d . Support Vector Machine (SVM)
model_name4 = "Support Vector Machine (SVM)"
# Note: probability=True is required to get predict_proba for ROC/AUC
svm_model = SVC(probability=True, random_state=42) 
svm_model.fit(x_train, y_train)
y_pred_svm = svm_model.predict(x_test)
y_proba_svm = svm_model.predict_proba(x_test)[:, 1]

if __name__ == '__main__':
    print(f'\nDataset split complete. Training samples: {x_train.shape[0]}, Testing samples: {x_test.shape[0]}')
    print(f'\nUsing the following {x_train.shape[1]} features: {list(x.columns)}\n')
    
    # regression data:
    evaluate_and_store(model_name1, lr_model, y_test, y_pred_lr, y_proba_lr)

    # decision tree:
    evaluate_and_store(model_name2, dt_model, y_test, y_pred_dt, y_proba_dt)

    # random forest
    evaluate_and_store(model_name3, rf_model, y_test, y_pred_rf, y_proba_rf)

    # support vector machine SVM
    evaluate_and_store(model_name4, svm_model, y_test, y_pred_svm, y_proba_svm)

    # 1. Print Combined Metrics Table
    evaluation_df = pd.DataFrame(final_metrics).T
    print("\n" + "="*70)
    print(f'Final Combined Model Evaluation')
    print("="*70)
    print(evaluation_df.sort_values(by='F1-score', ascending=False).to_markdown(floatfmt=".4f"))
    

    # writing the evaluated matrix into a file
    evaluation_df = pd.DataFrame(final_metrics).T
    metrics_text = evaluation_df.to_markdown(floatfmt=".4f")
    try:
        with open("evaluation_matrix.txt", 'w') as f:
            f.write("Final Baseline Model Evaluation:\n\n")
            f.write(metrics_text)
        print(f'Metrics saved to evaluation_matrix.txt')
    except Exception as e:
        print(f'Could not save file: {e}')

    # 2. Plot ROC Curve
    plt.figure(figsize=(10, 8))
    for name, data in roc_data.items():
        auc_score = evaluation_df.loc[name, 'AUC-score']
        plt.plot(data['fpr'], data['tpr'], label=f'{name} (AUC = {auc_score:.4f})')
    
    plt.plot([0, 1], [0, 1], 'r--', label='Baseline (AUC = 0.50)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()