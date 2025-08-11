#!/usr/bin/env python3
"""
Titanic Dataset - Exploratory Data Analysis (EDA)
Elevate Labs Data Analyst Internship - Task 5

Author: Data Analyst Intern
Date: August 2025
Description: Comprehensive EDA of Titanic dataset to extract insights using visual and statistical exploration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats

# Configure settings
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

def load_and_explore_data():
    """Load the Titanic dataset and perform initial exploration."""
    print("=" * 60)
    print("TITANIC DATASET - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Load dataset
    try:
        df = pd.read_csv('titanic_dataset.csv')
        print("✅ Dataset loaded successfully from local file!")
    except FileNotFoundError:
        print("📥 Loading dataset from online source...")
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
        print("✅ Dataset loaded successfully!")
    
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"📊 Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    return df

def basic_information(df):
    """Display basic dataset information."""
    print("\n" + "="*50)
    print("1. BASIC DATASET INFORMATION")
    print("="*50)
    
    print("\n🔍 Column Information:")
    print(df.info())
    
    print("\n🔍 First 5 rows:")
    print(df.head())
    
    print("\n🔍 Statistical Summary:")
    print(df.describe())
    
    return df

def missing_values_analysis(df):
    """Analyze missing values in the dataset."""
    print("\n" + "="*50)
    print("2. MISSING VALUES ANALYSIS")
    print("="*50)
    
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).values
    })
    
    print("\n🔍 Missing Values Summary:")
    print(missing_data)
    
    # Visualize missing values
    plt.figure(figsize=(10, 6))
    missing_data_viz = missing_data[missing_data['Missing_Count'] > 0]
    plt.bar(missing_data_viz['Column'], missing_data_viz['Missing_Percentage'], color='coral')
    plt.title('Missing Values Percentage by Column')
    plt.xlabel('Columns')
    plt.ylabel('Missing Percentage (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('missing_values_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def categorical_analysis(df):
    """Analyze categorical variables."""
    print("\n" + "="*50)
    print("3. CATEGORICAL VARIABLES ANALYSIS")
    print("="*50)
    
    categorical_columns = ['Survived', 'Pclass', 'Sex', 'Embarked']
    
    for col in categorical_columns:
        print(f"\n🔍 {col.upper()} - Value Counts:")
        print("-" * 30)
        value_counts = df[col].value_counts()
        print(value_counts)
        print("\nProportions:")
        print(df[col].value_counts(normalize=True).round(3))
    
    return df

def correlation_analysis(df):
    """Perform correlation analysis on numeric variables."""
    print("\n" + "="*50)
    print("4. CORRELATION ANALYSIS")
    print("="*50)
    
    # Select numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_columns].corr()
    
    print("\n🔍 Correlation Matrix:")
    print(correlation_matrix.round(3))
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, fmt='.3f')
    plt.title('Correlation Matrix - Titanic Dataset Variables', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation_matrix

def survival_analysis(df):
    """Analyze survival patterns by different categories."""
    print("\n" + "="*50)
    print("5. SURVIVAL ANALYSIS")
    print("="*50)
    
    # Overall survival rate
    survival_rate = df['Survived'].mean()
    print(f"\n📊 Overall Survival Rate: {survival_rate:.1%}")
    print(f"📊 Total Survivors: {df['Survived'].sum()}")
    print(f"📊 Total Deaths: {len(df) - df['Survived'].sum()}")
    
    # Survival by gender
    print("\n🔍 Survival Rate by Gender:")
    survival_by_sex = df.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean'])
    survival_by_sex.columns = ['Total_Count', 'Survived_Count', 'Survival_Rate']
    print(survival_by_sex)
    
    # Survival by class
    print("\n🔍 Survival Rate by Passenger Class:")
    survival_by_class = df.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean'])
    survival_by_class.columns = ['Total_Count', 'Survived_Count', 'Survival_Rate']
    print(survival_by_class)
    
    # Survival by embarkation
    print("\n🔍 Survival Rate by Embarkation Port:")
    survival_by_embarked = df.groupby('Embarked')['Survived'].agg(['count', 'sum', 'mean'])
    survival_by_embarked.columns = ['Total_Count', 'Survived_Count', 'Survival_Rate']
    print(survival_by_embarked)
    
    # Visualize survival rates
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Survival by Sex
    survival_by_sex['Survival_Rate'].plot(kind='bar', ax=axes[0], color='skyblue', alpha=0.8)
    axes.set_title('Survival Rate by Gender', fontsize=14)
    axes.set_ylabel('Survival Rate')
    axes.tick_params(axis='x', rotation=0)
    axes.grid(True, alpha=0.3)
    
    # Survival by Class
    survival_by_class['Survival_Rate'].plot(kind='bar', ax=axes[1], color='lightcoral', alpha=0.8)
    axes[1].set_title('Survival Rate by Class', fontsize=14)
    axes[1].set_ylabel('Survival Rate')
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].grid(True, alpha=0.3)
    
    # Survival by Embarkation
    survival_by_embarked['Survival_Rate'].plot(kind='bar', ax=axes[2], color='lightgreen', alpha=0.8)
    axes[2].set_title('Survival Rate by Embarkation', fontsize=14)
    axes[2].set_ylabel('Survival Rate')
    axes[2].tick_params(axis='x', rotation=0)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('survival_rates_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return survival_by_sex, survival_by_class, survival_by_embarked

def age_analysis(df):
    """Analyze age distribution and patterns."""
    print("\n" + "="*50)
    print("6. AGE ANALYSIS")
    print("="*50)
    
    # Age statistics
    print("\n🔍 Age Statistics:")
    age_stats = df['Age'].describe()
    print(age_stats)
    
    # Age skewness
    age_skewness = df['Age'].skew()
    print(f"\n📊 Age Skewness: {age_skewness:.3f}")
    
    # Create age groups
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                            labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
    
    print("\n🔍 Age Group Distribution:")
    print(df['Age_Group'].value_counts())
    
    print("\n🔍 Survival Rate by Age Group:")
    survival_by_age = df.groupby('Age_Group')['Survived'].agg(['count', 'sum', 'mean'])
    survival_by_age.columns = ['Total_Count', 'Survived_Count', 'Survival_Rate']
    print(survival_by_age)
    
    # Age distribution histogram
    plt.figure(figsize=(12, 6))
    plt.hist(df['Age'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    plt.title('Age Distribution of Titanic Passengers', fontsize=16)
    plt.xlabel('Age (years)')
    plt.ylabel('Frequency')
    plt.axvline(df['Age'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean Age: {df["Age"].mean():.1f}')
    plt.axvline(df['Age'].median(), color='green', linestyle='--', linewidth=2,
               label=f'Median Age: {df["Age"].median():.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def fare_analysis(df):
    """Analyze fare distribution and patterns."""
    print("\n" + "="*50)
    print("7. FARE ANALYSIS")
    print("="*50)
    
    # Fare statistics
    print("\n🔍 Fare Statistics:")
    fare_stats = df['Fare'].describe()
    print(fare_stats)
    
    # Fare skewness
    fare_skewness = df['Fare'].skew()
    print(f"\n📊 Fare Skewness: {fare_skewness:.3f}")
    
    # Create fare categories
    df['Fare_Category'] = pd.cut(df['Fare'], bins=[0, 10, 30, 100, 600], 
                                labels=['Low', 'Medium', 'High', 'Very High'])
    
    print("\n🔍 Fare Category Distribution:")
    print(df['Fare_Category'].value_counts())
    
    print("\n🔍 Survival Rate by Fare Category:")
    survival_by_fare = df.groupby('Fare_Category')['Survived'].agg(['count', 'sum', 'mean'])
    survival_by_fare.columns = ['Total_Count', 'Survived_Count', 'Survival_Rate']
    print(survival_by_fare)
    
    # Fare distribution by class - Box Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Pclass', y='Fare', palette='viridis')
    plt.title('Fare Distribution by Passenger Class', fontsize=16)
    plt.xlabel('Passenger Class')
    plt.ylabel('Fare (British Pounds)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fare_boxplot_by_class.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def family_analysis(df):
    """Analyze family size and survival patterns."""
    print("\n" + "="*50)
    print("8. FAMILY SIZE ANALYSIS")
    print("="*50)
    
    # Create family size variables
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
    df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)
    
    print("\n🔍 Family Size Distribution:")
    family_dist = df['Family_Size'].value_counts().sort_index()
    print(family_dist)
    
    print("\n🔍 Survival Rate by Family Size:")
    survival_by_family = df.groupby('Family_Size')['Survived'].agg(['count', 'sum', 'mean'])
    survival_by_family.columns = ['Total_Count', 'Survived_Count', 'Survival_Rate']
    print(survival_by_family)
    
    print("\n🔍 Alone vs With Family Survival:")
    survival_alone = df.groupby('Is_Alone')['Survived'].agg(['count', 'sum', 'mean'])
    survival_alone.columns = ['Total_Count', 'Survived_Count', 'Survival_Rate']
    survival_alone.index = ['With Family', 'Alone']
    print(survival_alone)
    
    return df

def comprehensive_visualizations(df):
    """Create comprehensive visualizations."""
    print("\n" + "="*50)
    print("9. COMPREHENSIVE VISUALIZATIONS")
    print("="*50)
    
    # Combined analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Age vs Survival
    sns.boxplot(data=df, x='Survived', y='Age', ax=axes[0,0], palette='Set2')
    axes[0,0].set_title('Age Distribution by Survival Status', fontsize=14)
    axes[0,0].set_xlabel('Survived (0=No, 1=Yes)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Fare vs Survival
    sns.boxplot(data=df, x='Survived', y='Fare', ax=axes[0,1], palette='Set2')
    axes[0,1].set_title('Fare Distribution by Survival Status', fontsize=14)
    axes[0,1].set_xlabel('Survived (0=No, 1=Yes)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Family Size vs Survival
    family_survival = df.groupby('Family_Size')['Survived'].mean()
    bars = axes[1,0].bar(family_survival.index, family_survival.values, color='coral', alpha=0.8)
    axes[1,0].set_title('Survival Rate by Family Size', fontsize=14)
    axes[1,0].set_xlabel('Family Size')
    axes[1,0].set_ylabel('Survival Rate')
    axes[1,0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.2f}', ha='center', va='bottom')
    
    # Sex and Class combined heatmap
    survival_pivot = df.pivot_table(values='Survived', index='Sex', columns='Pclass', aggfunc='mean')
    sns.heatmap(survival_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[1,1])
    axes[1,1].set_title('Survival Rate by Gender and Class', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ All visualizations saved successfully!")

def generate_insights(df):
    """Generate and display key insights."""
    print("\n" + "="*60)
    print("10. KEY INSIGHTS AND FINDINGS")
    print("="*60)
    
    insights = [
        "🔍 SURVIVAL OVERVIEW:",
        f"   • Overall survival rate: {df['Survived'].mean():.1%}",
        f"   • Total passengers: {len(df)}",
        f"   • Survivors: {df['Survived'].sum()}",
        f"   • Deaths: {len(df) - df['Survived'].sum()}",
        "",
        "👥 DEMOGRAPHIC PATTERNS:",
        f"   • Female survival rate: {df[df['Sex']=='female']['Survived'].mean():.1%}",
        f"   • Male survival rate: {df[df['Sex']=='male']['Survived'].mean():.1%}",
        f"   • Child survival rate: {df[df['Age'] <= 12]['Survived'].mean():.1%}",
        "",
        "💰 ECONOMIC FACTORS:",
        f"   • 1st Class survival: {df[df['Pclass']==1]['Survived'].mean():.1%}",
        f"   • 2nd Class survival: {df[df['Pclass']==2]['Survived'].mean():.1%}",
        f"   • 3rd Class survival: {df[df['Pclass']==3]['Survived'].mean():.1%}",
        "",
        "📊 DATA QUALITY:",
        f"   • Age missing: {df['Age'].isnull().sum()/len(df):.1%}",
        f"   • Cabin missing: {df['Cabin'].isnull().sum()/len(df):.1%}",
        f"   • Fare skewness: {df['Fare'].skew():.2f}",
        "",
        "🏆 KEY TAKEAWAYS:",
        "   • 'Women and children first' policy clearly evident",
        "   • Socioeconomic status significantly impacted survival",
        "   • Small families had better survival rates than large ones",
        "   • Higher fare passengers had better survival chances"
    ]
    
    for insight in insights:
        print(insight)

def main():
    """Main function to run the complete EDA."""
    print("🚢 Starting Titanic Dataset EDA Analysis...")
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Run all analysis functions
    df = basic_information(df)
    df = missing_values_analysis(df)
    df = categorical_analysis(df)
    correlation_matrix = correlation_analysis(df)
    survival_data = survival_analysis(df)
    df = age_analysis(df)
    df = fare_analysis(df)
    df = family_analysis(df)
    comprehensive_visualizations(df)
    generate_insights(df)
    
    print("\n" + "="*60)
    print("✅ EDA ANALYSIS COMPLETED SUCCESSFULLY!")
    print("📁 All visualizations have been saved as PNG files")
    print("📊 Check the generated charts and insights above")
    print("="*60)
    
    return df

if __name__ == "__main__":
    # Run the complete EDA analysis
    titanic_df = main()
