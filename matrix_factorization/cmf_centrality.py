"""
This module implements Collaborative Matrix Factorization (CMF) with centrality metrics.

The module provides functionality to evaluate the impact of network centrality metrics
on collaborative filtering recommendations. It includes methods for loading and
preprocessing centrality metrics, evaluating CMF models with user attributes, and
running comprehensive evaluations across different network models.

Main Components:
- Loading and preprocessing of centrality metrics from various network models
- Cross-validation evaluation of CMF with user attributes
- Comparative analysis between baseline CMF and CMF enhanced with centrality metrics
- Comprehensive evaluation across multiple network models (exponential, powerlaw, rayleigh)

The module supports different types of data transformations (scaling, normalization,
standardization) for centrality metrics and provides detailed evaluation metrics
including RMSE improvements and success rates across different network types.

Dependencies:
    - numpy
    - pandas
    - sklearn
    - cmfrec
    - tqdm

This module is part of a larger matrix factorization implementation and requires
additional utility functions from cmf.py and utils.py.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, LabelEncoder
from cmfrec import CMF
from tqdm import tqdm

from utils import load_dataset, split_data_single, predict_ratings
from cmf import search_best_params, evaluate_with_cv
from model_plots import plot_alpha_rmse_analysis


def load_centrality_metrics(model_name, network_index, transform='standardize'):
    """
    Load and preprocess centrality metrics from a CSV file.
    This function loads centrality metrics for a given model and network index from a CSV file,
    encodes user IDs, and applies the specified transformation to the metrics.
    Args:
        model_name : str
            Name of the model for which to load centrality metrics.
        network_index : int
            Index of the network for which to load centrality metrics.
        transform : str, optional. Literal['scale', 'normalize', 'standardize']
            Type of transformation to apply to the metrics. 
            Options are 'scale' (MinMaxScaler), 'normalize' (Normalizer), 
            or 'standardize' (StandardScaler). Default is 'standardize'.
    Returns:
        pandas.DataFrame or None
            DataFrame containing the transformed centrality metrics with encoded user IDs.
            Returns None if the file doesn't exist or if there's an error loading the data.
            The DataFrame has columns ['UserId', ...centrality_metrics].
    """
    data_path = f"../data/centrality_metrics/{model_name}"
    filename = f"centrality_metrics_{model_name}_{network_index:03d}.csv"
    filepath = os.path.join(data_path, filename)

    if not os.path.exists(filepath):
        return None

    try:
        centrality_df = pd.read_csv(filepath)
        user_encoder = LabelEncoder()
        centrality_df['UserId'] = user_encoder.fit_transform(centrality_df['UserId'])
        features = [col for col in centrality_df.columns if col != "UserId"]
        if transform == "scale":
            scaler = MinMaxScaler()
        elif transform == "normalize":
            scaler = Normalizer()
        elif transform == "standardize":
            scaler = StandardScaler()
        else:
            raise ValueError(
                "Invalid transform type. Choose from 'scale', 'normalize', or 'standardize'."
            )

        transformed = scaler.fit_transform(centrality_df[features])
        df_transformed = pd.concat(
            [
                centrality_df[["UserId"]].reset_index(drop=True),
                pd.DataFrame(transformed, columns=features)
            ],
            axis=1
        ).reset_index(drop=True)
        return df_transformed

    except Exception: # pylint: disable=broad-except
        return None

def evaluate_cmf_with_user_attributes(data, user_attributes, k, lambda_reg, method='als', n_splits=3):
    """
    Evaluate CMF with user attributes using cross-validation.
    Args:
        data (pd.DataFrame): DataFrame with columns ['UserId', 'ItemId', 'Rating']
        user_attributes (pd.DataFrame): DataFrame with columns ['UserId', ...centrality_metrics]
        k (int): Number of latent factors
        lambda_reg (float): Regularization parameter
        method (str): CMF method ('als' or 'lbfgs')
        n_splits (int): Number of cross-validation splits
    Returns:
        dict or None: Dictionary with mean RMSE, std RMSE, and number of successful splits,
                      or None if no successful evaluations.
    """
    rmse_scores = []

    for split in range(n_splits):
        try:
            # Split data
            train_data, test_data = split_data_single(data, test_size=0.2, random_state=42+split)

            # Get unique users in training set
            train_users = train_data['UserId'].unique()

            # Filter user attributes to only include training users
            available_attrs = user_attributes.loc[user_attributes['UserId'].isin(train_users)]

            if len(available_attrs) == 0:
                continue

            # Create user attributes matrix aligned with training users
            user_attr_matrix = available_attrs.reindex(train_users, fill_value=0.0)

            # Train CMF model with user attributes
            model = CMF(
                method=method,
                k=k,
                lambda_=lambda_reg,
                verbose=False,
                nthreads=-1
            )

            # Fit the model with user attributes
            model.fit(X=train_data, U=user_attr_matrix)

            # Predict on test set (only for users that have attributes)
            test_with_attrs = test_data[test_data['UserId'].isin(available_attrs.index)]

            if len(test_with_attrs) == 0:
                continue

            # Make predictions using the correct method signature
            test_predictions = predict_ratings(model, test_with_attrs)

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(test_data['Rating'], test_predictions))
            rmse_scores.append(rmse)

        except Exception: # pylint: disable=broad-except
            continue

    if len(rmse_scores) == 0:
        return None

    return {
        'mean_rmse': np.mean(rmse_scores),
        'std_rmse': np.std(rmse_scores),
        'n_splits': len(rmse_scores)
    }


def evaluate_single_network(data, model_name, network_index, best_params):
    """
    Evaluate impact of centrality metrics from a single network on CMF.
    Args:
        data (pd.DataFrame): DataFrame with columns ['UserId', 'ItemId', 'Rating']
        model_name (str): Name of the network model
        network_index (int): Index of the network
        best_params (dict): Best hyperparameters {'k': int, 'lambda_': float}
    Returns:
        dict or None: Dictionary with evaluation results or None if evaluation fails.
    """
    try:
        # Load centrality metrics
        user_attrs = load_centrality_metrics(model_name, network_index)

        # Baseline evaluation (without user attributes)
        baseline_result = evaluate_with_cv(
            data, k=best_params['k'],
            lambda_reg=best_params['lambda_'],
            n_splits=3, verbose=False
        )
        if baseline_result is None:
            return None

        # Enhanced evaluation (with user attributes)
        enhanced_result = evaluate_cmf_with_user_attributes(
            data,
            user_attrs,
            k=best_params['k'],
            lambda_reg=best_params['lambda_'],
            n_splits=3
        )
        if enhanced_result is None:
            return None

        # Calculate improvement
        baseline_rmse = baseline_result['rmse_mean']
        enhanced_rmse = enhanced_result['mean_rmse']
        improvement = ((baseline_rmse - enhanced_rmse) / baseline_rmse) * 100

        return {
            'model': model_name,
            'network_index': network_index,
            'baseline_rmse': baseline_rmse,
            'enhanced_rmse': enhanced_rmse,
            'improvement': improvement,
            'users_with_attributes': len(user_attrs)
        }

    except Exception: # pylint: disable=broad-except
        return None

def run_centrality_evaluation(sample_networks=5):
    """Run complete evaluation of centrality metrics across network models."""
    print("=" * 60)
    print("CENTRALITY METRICS EVALUATION FOR CMF")
    print("=" * 60)

    # Load dataset
    print("Loading dataset...")
    data = load_dataset('ratings_small.csv')
    if data is None:
        print("Failed to load dataset!")
        return None

    print(
        f"Dataset: {len(data)} ratings, "
        f"{data['UserId'].nunique()} users, "
        f"{data['ItemId'].nunique()} items"
    )

    # Find best hyperparameters
    print("Finding best hyperparameters...")
    search_result = search_best_params(data, n_iter=20, n_splits=3, verbose=False)
    if search_result is None:
        print("Hyperparameter search failed!")
        return None

    best_params = search_result['best_params']
    baseline_rmse = search_result['best_score']

    print(f"Best parameters: k={best_params['k']}, lambda={best_params['lambda_']:.6f}")
    print(f"Baseline RMSE: {baseline_rmse:.6f}")

    # Sample networks to evaluate
    models = ["exponential", "powerlaw", "rayleigh"]
    np.random.seed(42)
    network_indices = np.sort(np.random.choice(100, sample_networks, replace=False))

    print(f"Evaluating {sample_networks} networks from each model: {network_indices}")

    all_results = {'baseline_rmse': baseline_rmse, 'best_params': best_params, 'results': {}}

    for model_name in models:
        print(f"\n--- Evaluating {model_name.upper()} networks ---")
        model_results = []
        rmse_scores = []

        for net_idx in tqdm(network_indices, desc=f"{model_name.capitalize()}"):
            result = evaluate_single_network(data, model_name, net_idx, best_params)
            if result is not None:
                model_results.append(result)
                rmse_scores.append(result['enhanced_rmse'])
                print(f"  Network {net_idx}: {result['improvement']:+.3f}% improvement "
                      f"({result['users_with_attributes']} users)")
            else:
                print(f"  Network {net_idx}: Failed")

        if model_results:
            improvements = [r['improvement'] for r in model_results]
            avg_improvement = np.mean(improvements)
            positive_count = sum(1 for imp in improvements if imp > 0)

            print(f"\n{model_name.upper()} Summary:")
            print(f"  Successfully evaluated: {len(model_results)}/{sample_networks} networks")
            print(f"  Average improvement: {avg_improvement:+.3f}%")
            print(f"  Positive improvements: {positive_count}/{len(model_results)}")

            all_results['results'][model_name] = {
                'evaluations': model_results,
                'avg_improvement': avg_improvement,
                'positive_count': positive_count,
                'total_evaluated': len(model_results)
            }
        else:
            print(f"\n{model_name.upper()}: No successful evaluations")
            all_results['results'][model_name] = None

        plot_alpha_rmse_analysis(model_name, rmse_scores, baseline_rmse, save_plot=True)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Baseline RMSE: {baseline_rmse:.6f}")

    for model_name in models:
        result = all_results['results'][model_name]
        if result:
            print(f"{model_name.upper()}: {result['avg_improvement']:+.3f}% average improvement "
                  f"({result['positive_count']}/{result['total_evaluated']} positive)")
        else:
            print(f"{model_name.upper()}: No successful evaluations")

    return all_results

if __name__ == "__main__":
    results = run_centrality_evaluation(sample_networks=100)
    if results:
        print("\nEvaluation completed successfully!")
    else:
        print("\nEvaluation failed!")
