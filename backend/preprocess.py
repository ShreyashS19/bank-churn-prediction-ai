import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Handle missing values (if any)
    df = df.fillna(df.mean(numeric_only=True))
    
    # Fallback: if still NaN after mean (single-row case), fill with 0
    df = df.fillna(0)

    # Ensure correct column types
    categorical_cols = ['Gender', 'Education', 'Marital_Status', 'Income', 'Card_Category']
    
    numerical_cols = ['Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count', 
                     'Months_Inactive', 'Contacts_Count', 'Credit_Limit', 'Total_Revolving_Bal',
                     'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 
                     'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
    
    # Convert categorical columns to string
    # for col in categorical_cols:
    #     df[col] = df[col].astype(str)
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df