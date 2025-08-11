"""
Data Processing and Cleaning for Titanic Dataset
"""
import pandas as pd
import numpy as np

def clean_titanic_data():
    """Clean and preprocess the Titanic dataset."""
    
    # Load data
    df = pd.read_csv('titanic_dataset.csv')
    
    print("ORIGINAL DATA SUMMARY:")
    print("="*30)
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Handle missing values
    # Age: Fill with median by Sex and Pclass
    df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Embarked: Fill with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Cabin: Create binary feature
    df['Has_Cabin'] = (~df['Cabin'].isnull()).astype(int)
    
    # Feature engineering
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
    df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)
    
    # Extract title from name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    
    # Group rare titles
    rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    
    # Age groups
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 12, 18, 35, 60, 100],
                            labels=['Child', 'Teen', 'Young_Adult', 'Adult', 'Senior'])
    
    # Fare categories  
    df['Fare_Category'] = pd.qcut(df['Fare'], 
                                 q=4, 
                                 labels=['Low', 'Medium', 'High', 'Very_High'])
    
    print("\nCLEANED DATA SUMMARY:")
    print("="*30)
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Save cleaned data
    df.to_csv('titanic_cleaned.csv', index=False)
    print("\n✅ Cleaned data saved as 'titanic_cleaned.csv'")
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_titanic_data()
    print("\n📊 Data cleaning completed!")
