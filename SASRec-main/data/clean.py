import pandas as pd

def clean_dataset(df):
    # Handle missing values
    df = df.dropna(subset=['name', 'no_of_ratings'])

    # Clean 'no_of_ratings' by removing commas and converting to integer
    df['no_of_ratings'] = df['no_of_ratings'].replace({',': ''}, regex=True)
    df['no_of_ratings'] = pd.to_numeric(df['no_of_ratings'], errors='coerce')

    # Fill any NaNs in 'no_of_ratings' with the median
    df['no_of_ratings'] = df['no_of_ratings'].fillna(df['no_of_ratings'].median())
    
    return df
