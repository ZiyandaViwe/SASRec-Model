import pandas as pd
from collections import defaultdict
from data.clean import clean_dataset

def process_data(data_path, min_ratings=5):
    # Load and clean dataset
    data = clean_dataset(pd.read_csv(data_path))

    # Ensure 'no_of_ratings' is numeric (convert if necessary)
    data['no_of_ratings'] = pd.to_numeric(data['no_of_ratings'], errors='coerce')

    # Filter items with fewer than 'min_ratings'
    data = data[data['no_of_ratings'] >= min_ratings]

    # Extract relevant columns
    user_data = data[['name', 'no_of_ratings']]
    num_users = data['name'].nunique()
    num_items = len(data)  # Use total entries as item count for simplicity

    return user_data, num_users, num_items

# Test the function
data_path = '/home/sagemaker-user/SASRec-1/data/testdata.csv'
user_data, num_users, num_items = process_data(data_path)
