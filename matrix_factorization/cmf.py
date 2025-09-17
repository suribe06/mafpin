#!/usr/bin/env python3
"""
New CMF module combining the good RMSE results from old_cmf.py 
with the clean function structure from cmf.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from cmfrec import CMF


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
    Simple single split of data (like old_cmf.py uses in each iteration).
    
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


def train_model(train_data, k=20, lambda_reg=1.0, method="als", verbose=False):
    """
    Train CMF model exactly like old_cmf.py (which works well).
    
    Args:
        train_data (pd.DataFrame): Training data with UserId, ItemId, Rating
        k (int): Number of latent factors
        lambda_reg (float): Regularization parameter
        method (str): Optimization method
        verbose (bool): Print training progress
        
    Returns:
        CMF: Trained model or None if failed
    """
    try:
        # Parameter validation
        if k <= 0 or lambda_reg <= 0:
            raise ValueError(f"Invalid parameters: k={k}, lambda_reg={lambda_reg}")

        if len(train_data) == 0:
            raise ValueError("Training data is empty")

        # Initialize model exactly like old_cmf.py
        model = CMF(
            method=method,
            k=k,
            lambda_=lambda_reg,
            verbose=verbose
        )

        # CRITICAL: Fit with DataFrame directly, not numpy array
        model.fit(train_data)

        return model

    except Exception as e: # pylint: disable=broad-except
        if verbose:
            print(f"Training error: {str(e)}")
        return None


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


def evaluate_with_cv(data, k, lambda_reg, n_splits=10, test_size=0.2, verbose=False):
    """
    Evaluate model parameters using cross-validation.
    
    Args:
        data (pd.DataFrame): Complete dataset
        k (int): Number of latent factors
        lambda_reg (float): Regularization parameter
        n_splits (int): Number of train/test splits to average over
        test_size (float): Test set proportion
        verbose (bool): Print progress
        
    Returns:
        dict: Cross-validation results with mean and std of metrics
    """
    try:
        rmses = []
        maes = []
        r2s = []

        for _ in range(n_splits):
            # Random split (different each time, no fixed random_state)
            train_data, test_data = split_data_single(data, test_size=test_size, random_state=None)

            if train_data is None or test_data is None:
                continue

            # Train model
            model = train_model(train_data, k=k, lambda_reg=lambda_reg, verbose=False)
            if model is None:
                continue

            # Evaluate
            results = evaluate_single_split(model, test_data)
            if results is None:
                continue

            rmses.append(results['rmse'])
            maes.append(results['mae'])
            r2s.append(results['r2'])

        if not rmses:
            return None

        # Calculate statistics
        mean_rmse = np.mean(rmses)
        std_rmse = np.std(rmses)
        mean_mae = np.mean(maes)
        std_mae = np.std(maes)
        mean_r2 = np.mean(r2s)
        std_r2 = np.std(r2s)

        results = {
            'rmse_mean': mean_rmse,
            'rmse_std': std_rmse,
            'mae_mean': mean_mae,
            'mae_std': std_mae,
            'r2_mean': mean_r2,
            'r2_std': std_r2,
            'n_successful_splits': len(rmses)
        }

        if verbose:
            print(f"CV Results (k={k}, lambda={lambda_reg:.4f}):")
            print(f"  RMSE: {mean_rmse:.6f} (+/- {std_rmse:.6f})")
            print(f"  MAE:  {mean_mae:.6f} (+/- {std_mae:.6f})")
            print(f"  R²:   {mean_r2:.6f} (+/- {std_r2:.6f})")
            print(f"  Successful splits: {len(rmses)}/{n_splits}")

        return results

    except Exception as e: # pylint: disable=broad-except
        print(f"Cross-validation error: {str(e)}")
        return None


def search_best_params(data, param_grid=None, n_iter=50, n_splits=10, test_size=0.2, verbose=True):
    """
    Hyperparameter search using cross-validation (like old_cmf.py approach).
    
    Args:
        data (pd.DataFrame): Complete dataset
        param_grid (dict): Parameter distributions
        n_iter (int): Number of random parameter combinations
        n_splits (int): Number of CV splits per parameter combination
        test_size (float): Test set proportion
        verbose (bool): Print progress
        
    Returns:
        dict: Best parameters and all results
    """
    try:
        # Default parameter grid
        if param_grid is None:
            param_grid = {
                'k': randint(10, 101),        # 10 to 100 factors
                'lambda_': uniform(1, 50),    # 1 to 50 (uniform)
            }

        print("Starting randomized hyperparameter search with cross-validation...")
        print(f"Iterations: {n_iter}, CV splits per iteration: {n_splits}")

        best_score = float('inf')
        best_params = None
        all_results = []

        # Set seed for reproducibility
        np.random.seed(42)

        for i in range(n_iter):
            # Sample parameters
            k = int(param_grid['k'].rvs())
            lambda_reg = float(param_grid['lambda_'].rvs())

            current_params = {'k': k, 'lambda_': lambda_reg}

            # Cross-validation evaluation
            cv_results = evaluate_with_cv(
                data, k, lambda_reg,
                n_splits=n_splits,
                test_size=test_size,
                verbose=False
            )

            if cv_results is None:
                if verbose:
                    print(f"Iter {i+1:3d}: k={k:2d}, lambda={lambda_reg:6.4f} -> FAILED")
                continue

            mean_rmse = cv_results['rmse_mean']
            std_rmse = cv_results['rmse_std']

            # Store results
            result_entry = current_params.copy()
            result_entry.update(cv_results)
            result_entry['iteration'] = i + 1
            all_results.append(result_entry)

            # Update best
            if mean_rmse < best_score:
                best_score = mean_rmse
                best_params = current_params.copy()

            if verbose:
                status = "***" if mean_rmse == best_score else "   "
                print(f"Iter {i+1:3d}: k={k:2d}, lambda={lambda_reg:6.4f} -> "
                      f"RMSE={mean_rmse:.6f} (+/- {std_rmse:.6f}) {status}")

        # Summary
        if best_params is not None:
            print(f"\n{'='*70}")
            print("BEST PARAMETERS FOUND:")
            print(f"{'='*70}")
            print(f"Best RMSE: {best_score:.6f}")
            print(f"Best k: {best_params['k']}")
            print(f"Best lambda: {best_params['lambda_']:.6f}")
            print(f"{'='*70}")

            # Top 5 results
            if all_results:
                sorted_results = sorted(all_results, key=lambda x: x['rmse_mean'])[:5]
                print("\nTop 5 parameter combinations:")
                for i, result in enumerate(sorted_results, 1):
                    print(f"  {i}. k={result['k']:2d}, lambda={result['lambda_']:6.4f}, "
                          f"RMSE={result['rmse_mean']:.6f} (+/- {result['rmse_std']:.6f})")
        else:
            print("No valid parameters found!")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results,
            'n_iterations': n_iter,
            'cv_splits': n_splits
        }

    except Exception as e: # pylint: disable=broad-except
        print(f"Parameter search error: {str(e)}")
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


def train_final_model(data, best_params, verbose=True):
    """
    Train final model with best parameters on full dataset.
    
    Args:
        data (pd.DataFrame): Full dataset
        best_params (dict): Best parameters from search
        verbose (bool): Print progress
        
    Returns:
        CMF: Trained model
    """
    try:
        print("Training final model with best parameters...")
        print(f"Parameters: k={best_params['k']}, lambda={best_params['lambda_']:.6f}")

        model = train_model(
            data,
            k=best_params['k'],
            lambda_reg=best_params['lambda_'],
            verbose=verbose
        )

        if model is not None:
            print("Final model trained successfully!")
        else:
            print("Failed to train final model!")

        return model

    except Exception as e: # pylint: disable=broad-except
        print(f"Final training error: {str(e)}")
        return None


def run_complete_example(filename='ratings_small.csv'):
    """
    Complete example workflow matching old_cmf.py results.
    
    Args:
        filename (str): Data file name
    """
    print("=" * 80)
    print("CMF: COMPLETE WORKFLOW")
    print("=" * 80)

    # 1. Load data
    print("\n1. Loading and preprocessing data...")
    data = load_dataset(filename)
    if data is None:
        print("Failed to load data!")
        return

    # 2. Hyperparameter search
    print("\n2. Hyperparameter search with cross-validation...")
    search_results = search_best_params(
        data,
        n_iter=50,
        n_splits=10,
        test_size=0.2,
        verbose=True
    )

    if search_results is None or search_results['best_params'] is None:
        print("Hyperparameter search failed!")
        return

    best_params = search_results['best_params']
    best_rmse = search_results['best_score']

    print("\nFinal Results:")
    print(f"Best RMSE: {best_rmse:.6f}")
    print(f"Best parameters: k={best_params['k']}, lambda={best_params['lambda_']:.6f}")

    # 3. Train final model
    print("\n3. Training final model...")
    final_model = train_final_model(data, best_params, verbose=True)

    if final_model is None:
        print("Failed to train final model!")
        return

    # 4. Generate sample recommendations
    print("\n4. Generating sample recommendations...")
    sample_user = 0  # First user (encoded)
    recommendations = make_recommendations(final_model, sample_user, data, n_recommendations=5)

    if recommendations:
        print(f"Top 5 recommendations for user {sample_user}:")
        for i, (item_id, rating) in enumerate(recommendations, 1):
            print(f"  {i}. Item {item_id}: {rating:.4f}")

    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    return {
        'model': final_model,
        'best_params': best_params,
        'best_rmse': best_rmse,
        'search_results': search_results
    }

if __name__ == "__main__":
    # Run the complete example
    results = run_complete_example('ratings_small.csv')
