# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (replace 'loan_data.csv' with the actual filename)
# Make sure the CSV file is in the same directory or provide the full path
data = pd.read_csv('loan_data.csv')

# Display the first few rows of the dataset to understand its structure
print("First few rows of the dataset:")
print(data.head())

# Display basic information about the dataset (data types, missing values, etc.)
print("\nDataset information:")
print(data.info())

# Get a statistical summary of the numerical features
print("\nStatistical summary of numerical features:")
print(data.describe())

# Check for missing values in each column
print("\nMissing values in each column:")
print(data.isnull().sum())

# Check the distribution of the target variable (Loan Status)
print("\nDistribution of Loan Status (target variable):")
print(data['Loan_Status'].value_counts())

# Visualize the target variable distribution
sns.countplot(x='Loan_Status', data=data)
plt.title('Loan Status Distribution')
plt.show()

# Explore correlations between numerical features and the target variable
# Convert 'Loan_Status' into numerical values (e.g., 1 for Approved, 0 for Not Approved)
data['Loan_Status_numeric'] = data['Loan_Status'].map({'Y': 1, 'N': 0})

# Compute the correlation matrix
correlation_matrix = data.corr(numeric_only=True)

# Display the correlation matrix for a quick overview of relationships
print("\nCorrelation matrix:")
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Explore categorical features (e.g., gender, education, property area)
# Display the unique values for some key categorical columns
categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
for column in categorical_columns:
    print(f"\nUnique values in '{column}' column:")
    print(data[column].value_counts())

# Visualize the relationship between a categorical feature and loan status (e.g., Gender and Loan Status)
sns.countplot(x='Gender', hue='Loan_Status', data=data)
plt.title('Gender vs Loan Status')
plt.show()

sns.countplot(x='Education', hue='Loan_Status', data=data)
plt.title('Education vs Loan Status')
plt.show()

