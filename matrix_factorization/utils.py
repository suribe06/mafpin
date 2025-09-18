"""
Utility functions for matrix factorization models.
This module provides utility functions for loading, preprocessing, and evaluating
matrix factorization models for recommendation systems. Key functionality includes:
- Loading and preprocessing rating datasets
- Encoding user and item IDs to contiguous integers 
- Splitting data into train/test sets
- Making predictions using trained models
- Evaluating model performance with metrics like RMSE, MAE and R-squared
The functions are designed to work with collaborative matrix factorization (CMF)
models but can be used with other matrix factorization approaches as well.
Functions:
    load_dataset: Load and preprocess ratings data from CSV
    split_data_single: Split data into train/test sets
    predict_ratings: Generate predictions from trained model
    evaluate_single_split: Calculate evaluation metrics on test data
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_dataset(filename):
    """
    Load dataset from CSV file with proper preprocessing.
    
    Args:
        filename (str): Name of the CSV file in ../data/ directory
        
    Returns:
        pd.DataFrame: DataFrame with properly encoded UserId, ItemId, Rating
    """
    try:
        filepath = os.path.join('..', 'data', filename)
        if not os.path.exists(filepath):
            print(f"Error: File '{filename}' not found at {filepath}")
            return None

        # Load the dataset - use first 3 columns regardless of names
        data = pd.read_csv(filepath, usecols=[0, 1, 2])
        data.columns = ['UserId', 'ItemId', 'Rating']  # Standardize column names

        print("Dataset loaded successfully:")
        print(f"  - Shape: {data.shape}")
        print(f"  - Original Users: {data['UserId'].nunique()}")
        print(f"  - Original Items: {data['ItemId'].nunique()}")
        print(f"  - Ratings: {len(data)}")
        print(f"  - Rating range: [{data['Rating'].min()}, {data['Rating'].max()}]")

        # CRITICAL: Encode UserId and ItemId to contiguous integers starting from 0
        # This is essential for CMF to work properly!
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        data['UserId'] = user_encoder.fit_transform(data['UserId'])
        data['ItemId'] = item_encoder.fit_transform(data['ItemId'])

        print("After encoding:")
        print(f"  - Encoded Users: 0 to {data['UserId'].max()}")
        print(f"  - Encoded Items: 0 to {data['ItemId'].max()}")

        return data

    except Exception as e: # pylint: disable=broad-except
        print(f"Error loading dataset: {str(e)}")
        return None


def split_data_single(data, test_size=0.2, random_state=None):
    """
    Simple single split of data.
    
    Args:
        data (pd.DataFrame): Complete dataset
        test_size (float): Proportion for test set
        random_state (int): Random seed (None for random)
        
    Returns:
        tuple: (train_data, test_data)
    """
    try:
        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state
        )
        return train_data, test_data

    except Exception as e: # pylint: disable=broad-except
        print(f"Error splitting data: {str(e)}")
        return None, None

def predict_ratings(model, test_data):
    """
    Predict ratings using the same method as old_cmf.py.
    
    Args:
        model: Trained CMF model
        test_data (pd.DataFrame): Test data with UserId, ItemId columns
        
    Returns:
        np.array: Predicted ratings or None if failed
    """
    try:
        if model is None:
            return None
        if len(test_data) == 0:
            print("Warning: Test data is empty")
            return np.array([])

        predictions = model.predict(test_data['UserId'], test_data['ItemId'])

        return np.array(predictions)

    except Exception as e: # pylint: disable=broad-except
        print(f"Prediction error: {str(e)}")
        return None

def evaluate_single_split(model, test_data):
    """
    Evaluate model on a single test split.
    
    Args:
        model: Trained CMF model
        test_data (pd.DataFrame): Test data
        
    Returns:
        dict: Evaluation metrics or None if failed
    """
    try:
        predictions = predict_ratings(model, test_data)
        if predictions is None:
            return None

        actual = test_data['Rating'].values

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)

        # R-squared
        ss_res = np.sum((actual - predictions) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_predictions': len(predictions)
        }

    except Exception as e: # pylint: disable=broad-except
        print(f"Evaluation error: {str(e)}")
        return None

def make_recommendations(model, user_id, data, n_recommendations=10):
    """
    Generate recommendations for a user.
    
    Args:
        model: Trained CMF model
        user_id (int): Encoded user ID
        data (pd.DataFrame): Original data to exclude seen items
        n_recommendations (int): Number of recommendations
        
    Returns:
        list: List of (item_id, predicted_rating) tuples
    """
    try:
        if model is None:
            return None

        # Get all items
        all_items = np.arange(data['ItemId'].max() + 1)

        # Get items already rated by user
        seen_items = set(data[data['UserId'] == user_id]['ItemId'].values)

        # Candidate items (not yet rated)
        candidate_items = [item for item in all_items if item not in seen_items]

        if not candidate_items:
            print(f"No new items to recommend for user {user_id}")
            return []

        # Create test data for predictions
        test_df = pd.DataFrame({
            'UserId': [user_id] * len(candidate_items),
            'ItemId': candidate_items,
            'Rating': [0] * len(candidate_items)  # Dummy ratings
        })

        # Get predictions
        predictions = predict_ratings(model, test_df)
        if predictions is None:
            return None

        # Combine and sort
        recommendations = list(zip(candidate_items, predictions))
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations[:n_recommendations]

    except Exception as e: # pylint: disable=broad-except
        print(f"Recommendation error: {str(e)}")
        return None
