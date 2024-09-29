import seaborn as sns
import matplotlib.pyplot as plt

def plot_churn_distribution(df):
    sns.countplot(data=df, x='Churn')
    plt.title('Churn Distribution')
    plt.show()

def plot_numerical_distributions(df, numerical_features):
    df[numerical_features].hist(bins=15, figsize=(15, 6), layout=(2, 4))
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 8))
    #only numbers column
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()