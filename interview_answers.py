"""
EDA Interview Questions - Answers with Code Examples
Elevate Labs Data Analyst Internship
"""

print("EDA INTERVIEW QUESTIONS - PRACTICAL ANSWERS")
print("="*50)

# Load data for examples
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv('titanic_dataset.csv')

print("\n1. What is EDA and why is it important?")
print("-" * 40)
print("""
EDA (Exploratory Data Analysis) is the process of analyzing datasets to summarize 
their main characteristics using statistical graphics and other data visualization methods.

Importance:
• Understand data structure and quality
• Identify patterns, trends, and relationships
• Detect outliers and anomalies
• Guide hypothesis formation
• Inform feature selection and model choice
• Validate data assumptions
""")

print("\n2. Which plots do you use to check correlation?")
print("-" * 40)
print("Demonstration with Titanic data:")

# Correlation matrix
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.savefig('correlation_demo.png')
plt.show()

# Pairplot
sns.pairplot(df[['Age', 'Fare', 'SibSp', 'Parch', 'Survived']], hue='Survived')
plt.savefig('pairplot_demo.png')
plt.show()

print("Plots used: Heatmap, Pairplot, Scatter plot")

print("\n3. How do you handle skewed data?")
print("-" * 40)
print(f"Fare skewness: {df['Fare'].skew():.2f}")

# Show skewness handling
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
df['Fare'].hist(bins=30)
plt.title(f'Original Fare (Skew: {df["Fare"].skew():.2f})')

plt.subplot(1, 3, 2)
log_fare = np.log1p(df['Fare'])
log_fare.hist(bins=30)
plt.title(f'Log Transformed (Skew: {log_fare.skew():.2f})')

plt.subplot(1, 3, 3)
sqrt_fare = np.sqrt(df['Fare'])
sqrt_fare.hist(bins=30)
plt.title(f'Square Root (Skew: {sqrt_fare.skew():.2f})')

plt.tight_layout()
plt.savefig('skewness_handling.png')
plt.show()

print("""
Methods to handle skewed data:
• Log transformation: log(x+1)
• Square root transformation: sqrt(x)  
• Box-Cox transformation
• Yeo-Johnson transformation
• Quantile transformation
""")

print("\n4. How to detect multicollinearity?")
print("-" * 40)

# VIF calculation
def calculate_vif(X):
    vif = pd.DataFrame()
    vif["Feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

numeric_features = df[['Age', 'SibSp', 'Parch', 'Fare']].dropna()
vif_df = calculate_vif(numeric_features)
print("VIF Values:")
print(vif_df)

print("""
Detection methods:
• VIF (Variance Inflation Factor) > 5 or 10
• Correlation matrix |r| > 0.7-0.9  
• Condition Index > 30
• Eigenvalues close to 0
""")

print("\n5. What are univariate, bivariate, and multivariate analyses?")
print("-" * 40)

print("UNIVARIATE ANALYSIS (Single variable):")
print(f"Age mean: {df['Age'].mean():.1f}")
print(f"Age median: {df['Age'].median():.1f}")
print(f"Age std: {df['Age'].std():.1f}")

print("\nBIVARIATE ANALYSIS (Two variables):")
age_survival_corr = df[['Age', 'Survived']].corr().iloc[0,1]
print(f"Age-Survival correlation: {age_survival_corr:.3f}")

print("\nMULTIVARIATE ANALYSIS (Multiple variables):")
print("Multiple regression, PCA, clustering, etc.")

print("\n6. Difference between heatmap and pairplot?")
print("-" * 40)
print("""
HEATMAP:
• Shows correlation coefficients numerically
• Color-coded matrix format
• Good for many variables at once
• Shows only linear relationships (Pearson correlation)

PAIRPLOT:  
• Shows actual scatter plots for each variable pair
• Distributions on diagonal
• Can reveal non-linear relationships
• Better for detailed pair-wise analysis
• More space-consuming
""")

print("\n7. How do you summarize your insights?")
print("-" * 40)
print("""
Structure for summarizing EDA insights:

1. KEY FINDINGS (Executive Summary)
   • Most important discoveries
   • Business impact/implications

2. DATA QUALITY ASSESSMENT
   • Missing values, outliers, inconsistencies
   • Data completeness and reliability

3. UNIVARIATE INSIGHTS
   • Distribution characteristics
   • Central tendencies and spread
   • Skewness and outliers

4. BIVARIATE/MULTIVARIATE RELATIONSHIPS
   • Strong correlations
   • Interesting patterns
   • Unexpected findings

5. ACTIONABLE RECOMMENDATIONS
   • Data preprocessing steps needed
   • Feature engineering opportunities
   • Modeling considerations

6. LIMITATIONS AND ASSUMPTIONS
   • What the data can/cannot tell us
   • Potential biases or gaps
""")

print("\nPRACTICAL EXAMPLE FROM TITANIC DATA:")
print("="*50)
print("Key Finding: Gender was the strongest survival predictor")
print(f"• Female survival: {df[df['Sex']=='female']['Survived'].mean():.1%}")
print(f"• Male survival: {df[df['Sex']=='male']['Survived'].mean():.1%}")
print("• Recommendation: Include gender as primary feature in model")
print("• Business insight: 'Women and children first' policy was followed")

