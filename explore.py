# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset using pandas
# Assuming 'diabetes_dataset.csv' is your dataset file
dataset_path = 'diabetes.csv'
df = pd.read_csv(dataset_path)

# Display basic information about the dataset
print(df.info())

# Display the first few rows of the dataset
print(df.head())

# Pairplot: Visualize relationships between numerical features
sns.pairplot(df, hue='Outcome', diag_kind='kde')
plt.show()

# Correlation Matrix: Explore correlation between features
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Histograms: Visualize distribution of individual features
df.hist(figsize=(12, 10), bins=20)
plt.suptitle("Histogram of Features")
plt.show()

# Boxplots: Identify outliers in numerical features
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Outcome', y=feature, data=df)
    plt.title(f'Boxplot of {feature}')
    plt.show()
