# Simple EDA Script for Quick Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('titanic_dataset.csv')

# Basic info
print("Dataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Survival analysis
print(f"\nSurvival Rate: {df['Survived'].mean():.1%}")
print("\nSurvival by Gender:")
print(df.groupby('Sex')['Survived'].mean())

# Quick visualizations
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
df['Age'].hist(bins=30)
plt.title('Age Distribution')

plt.subplot(2, 3, 2)
sns.countplot(data=df, x='Survived')
plt.title('Survival Count')

plt.subplot(2, 3, 3)
sns.countplot(data=df, x='Pclass')
plt.title('Class Distribution')

plt.subplot(2, 3, 4)
survival_by_class = df.groupby('Pclass')['Survived'].mean()
survival_by_class.plot(kind='bar')
plt.title('Survival by Class')

plt.subplot(2, 3, 5)
survival_by_sex = df.groupby('Sex')['Survived'].mean()
survival_by_sex.plot(kind='bar')
plt.title('Survival by Gender')

plt.subplot(2, 3, 6)
sns.boxplot(data=df, x='Pclass', y='Fare')
plt.title('Fare by Class')

plt.tight_layout()
plt.savefig('quick_eda_analysis.png', dpi=300)
plt.show()
