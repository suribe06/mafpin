"""
Cascade Generation Script

This script generates cascades from rating datasets and saves them to a cascades.txt file.

Expected Input Format:
The input CSV file should have the following columns (in order):
- UserId: Unique identifier for users
- ItemId: Unique identifier for items (movies, products, etc.)
- Rating: Rating given by user to item
- timestamp: Unix timestamp of when the rating was given

Output Format:
The output cascades.txt file contains:
1. User mappings (user_id,user_id format)
2. Empty line separator
3. Cascade data with a line per cascade, line i corresponds to item_node i
(comma-separated: user_node, timestamp, user_node, timestamp, ...)

Usage:
- python generate_cascades.py [dataset_name]
- python generate_cascades.py --help

Example:
- python generate_cascades.py ratings_small
"""

import datetime
import itertools
import os
import sys

import numpy as np
import pandas as pd

def list_available_datasets():
    """
    List all CSV files available in the ./data/ folder.
    
    Returns:
        list: List of dataset names (without .csv extension)
    """
    data_folder = os.path.join('..', 'data')
    if not os.path.exists(data_folder):
        print(f"Error: Data folder '{data_folder}' not found.")
        return []

    available_datasets = []
    for file in os.listdir(data_folder):
        if file.endswith('.csv'):
            available_datasets.append(file[:-4])  # Remove .csv extension

    return available_datasets

def generate_cascades(input_dataset):
    """
    Generate cascades from a dataset and save to cascades.txt file.
    
    Args:
        input_dataset (str): Name of the dataset file (without extension) in the ./data/ folder
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Construct the data file path
    data_file = os.path.join('..', 'data', f'{input_dataset}.csv')
    output_file = os.path.join('..', 'data', 'cascades.txt')

    try:
        # Check if the dataset file exists
        if not os.path.exists(data_file):
            print(f"Error: Dataset file '{data_file}' not found.")
            return False

        print(f"Processing dataset: {data_file}")

        # Load dataset (assumes first 4 columns: UserId, ItemId, Rating, Timestamp)
        interactions = pd.read_csv(data_file, usecols=[0, 1, 2, 3])

        # Get the number of unique users and items
        num_users = interactions["UserId"].nunique()
        num_items = interactions["ItemId"].nunique()

        print(f"Found {num_items} unique items and {num_users} unique users")

        # Create mappings for item and user IDs
        item_mapper = dict(zip(np.unique(interactions["ItemId"]), range(num_items)))
        user_mapper = dict(
            zip(np.unique(interactions["UserId"]), range(num_items, num_items + num_users))
        )

        # Initialize cascades (key=item_id, value=list of [user, timestamp])
        cascades = {item_id: [] for item_id in item_mapper.values()}

        # Build cascades
        for _, row in interactions.iterrows():
            item_id = item_mapper[row["ItemId"]]
            user_id = user_mapper[row["UserId"]]
            timestamp = datetime.datetime.fromtimestamp(row["timestamp"])
            cascades[item_id].append([user_id, timestamp])

        # Sort cascades (descending) and convert timestamps to Unix format
        for item_id, records in cascades.items():
            records.sort(key=lambda x: x[1], reverse=True)
            cascades[item_id] = list(itertools.chain(*[[u, dt.timestamp()] for u, dt in records]))

        # Write cascades to file
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write user mapper values first
            for u in list(user_mapper.values()):
                f.write(f"{u},{u}\n")

            # Write empty line
            f.write("\n")

            # Write cascades
            for record in cascades.values():
                if record: # Only write non-empty cascades
                    f.write(",".join(map(str, record)) + "\n")

        print(f"Cascades successfully generated and saved to: {output_file}")
        return True

    except Exception as e: # pylint: disable=broad-except
        print(f"Error processing dataset: {str(e)}")
        return False

if __name__ == '__main__':
    # Check if help is requested
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print("Usage: python generate_cascades.py [dataset_name]")
        print("\nAvailable datasets:")
        datasets = list_available_datasets()
        if datasets:
            for dataset in datasets:
                print(f"  - {dataset}")
        else:
            print("  No CSV files found in ./data/ folder")
        print("\nExample: python generate_cascades.py ratings_small")
        sys.exit(0)

    # Check if dataset name is provided as command line argument
    DATASET_NAME = ""
    if len(sys.argv) > 1:
        DATASET_NAME = sys.argv[1]
    else:
        # Show available datasets and use default
        datasets = list_available_datasets()
        if datasets:
            print("Available datasets:")
            for dataset in datasets:
                print(f"  - {dataset}")
            print()
        DATASET_NAME = "ratings_small"  # Default dataset name

    print(f"Using dataset: {DATASET_NAME}")
    SUCCESS = generate_cascades(DATASET_NAME)

    if not SUCCESS:
        print("Cascade generation failed.")
