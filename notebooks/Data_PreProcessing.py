import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# 1 - Load the Heart Disease UCI dataset into a Pandas DataFrame 
df = pd.read_csv('heart.csv',
                 na_values = [' ' , 'none'],)

cols_to_convert =  ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col] , errors='coerce')

intial_rows = len(df)
# 2 - Handle missing values (removal).
df_cleaned = df.dropna(axis = 0, how='any')
final_rows = len(df_cleaned)

# 3 - Perform data encoding (one-hot encoding for categorical variables).
df_encoded = pd.get_dummies(df_cleaned , 
                             columns = cols_to_convert,
                             drop_first = True)

numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# 4 - Standardize numerical features using MinMaxScaler.
scaler = MinMaxScaler() 
df_scaled = df_encoded.copy()
df_scaled.loc[:,numerical_cols] = scaler.fit_transform(df_scaled.loc[:, numerical_cols]) 

if __name__ == '__main__':
    print(f'Total rows before removal: \n{intial_rows}')
    print(f'Missing Values Before Handling: \n{df.isnull().sum()}')
    print(f'Standardization complete')
    print(f'Descriptive statistics of scaled features: \n{df_scaled[numerical_cols].describe().T}')
    print(f'Missing values after removal: \n{df_cleaned.isnull().sum()}')
    print(f'Final total number of rows after removal:  {final_rows}')

    print(f'Number of rows removed:  {intial_rows - final_rows}')
    print(f'Data Encoding Results\nOriginal Columns:  {df_cleaned.shape[1]}\nEncoded Columns:  {df_encoded.shape[1]}')
    print("\nNew columns created:")
    print([col for col in df_encoded.columns if any(c in col for c in cols_to_convert)])
    print(df_cleaned.nunique())

    # 5 -Conduct Exploratory Data Analysis (EDA)
    # a - with histograms
    plt.figure(figsize=(15,10))
    df_scaled[numerical_cols].hist(bins = 20 , ax=plt.gca())
    plt.suptitle('Histograms of Standardized Numerical Features' , y=1.02)
    plt.show()

    # b - with correlation heatmap
    corr_matrix = df_scaled.corr()
    plt.figure(figsize=(18,15))
    corr = corr_matrix['target'].sort_values(ascending = False).index
    sns.heatmap(corr_matrix.loc[corr , corr],
            annot = False,
            linewidths = .5)
    plt.title('Correlation Heatmap of All Feature')
    plt.show()

    # c - with boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data = df_scaled[numerical_cols])
    plt.title('Boxplots of Standardized Numerical Features (Outlier Check)')
    plt.show()