#!/usr/bin/env python3
"""
Model Plots Module for Collaborative Matrix Factorization

This module contains visualization functions for analyzing hyperparameter search results,
metrics comparison, and convergence analysis from CMF model training.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_hyperparameter_search_results(search_results, save_path=None, figsize=(15, 10)):
    """
    Plot comprehensive visualization of hyperparameter search results.
    
    Args:
        search_results (dict): Results from search_best_params function
        save_path (str, optional): Path to save the plot
        figsize (tuple): Figure size (width, height)
    """
    try:
        if search_results is None or 'all_results' not in search_results:
            print("Error: No search results to plot")
            return

        results = search_results['all_results']
        if not results:
            print("Error: No results data to plot")
            return

        # Extract data
        k_values = [r['k'] for r in results]
        lambda_values = [r['lambda_'] for r in results]
        rmse_values = [r['rmse_mean'] for r in results]
        rmse_std_values = [r['rmse_std'] for r in results]
        mae_values = [r['mae_mean'] for r in results]
        r2_values = [r['r2_mean'] for r in results]

        # Create subplot layout
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Hyperparameter Search Results Analysis', fontsize=16, fontweight='bold')

        # 1. RMSE vs k
        axes[0, 0].scatter(k_values, rmse_values, alpha=0.6, c='blue')
        axes[0, 0].set_xlabel('Number of Factors (k)')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_title('RMSE vs Number of Factors')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. RMSE vs lambda
        axes[0, 1].scatter(lambda_values, rmse_values, alpha=0.6, c='red')
        axes[0, 1].set_xlabel('Regularization (λ)')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('RMSE vs Regularization')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. RMSE distribution
        axes[0, 2].hist(rmse_values, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 2].axvline(search_results['best_score'], color='red', linestyle='--',
                          label=f"Best: {search_results['best_score']:.4f}")
        axes[0, 2].set_xlabel('RMSE')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('RMSE Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. MAE vs RMSE
        axes[1, 0].scatter(rmse_values, mae_values, alpha=0.6, c='purple')
        axes[1, 0].set_xlabel('RMSE')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('MAE vs RMSE Correlation')
        axes[1, 0].grid(True, alpha=0.3)

        # 5. R² vs RMSE
        axes[1, 1].scatter(rmse_values, r2_values, alpha=0.6, c='orange')
        axes[1, 1].set_xlabel('RMSE')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].set_title('R² vs RMSE Correlation')
        axes[1, 1].grid(True, alpha=0.3)

        # 6. RMSE with error bars (top 20 results)
        sorted_results = sorted(results, key=lambda x: x['rmse_mean'])[:20]
        x_pos = range(len(sorted_results))
        rmse_means = [r['rmse_mean'] for r in sorted_results]
        rmse_stds = [r['rmse_std'] for r in sorted_results]

        axes[1, 2].errorbar(x_pos, rmse_means, yerr=rmse_stds,
                           fmt='o', capsize=3, alpha=0.7)
        axes[1, 2].set_xlabel('Top Parameter Combinations (ranked)')
        axes[1, 2].set_ylabel('RMSE')
        axes[1, 2].set_title('Top 20 Results with Error Bars')
        axes[1, 2].grid(True, alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save if requested
        if save_path:
            path = os.path.join("..", "plots", save_path)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {path}")

        plt.show()

    except Exception as e: # pylint: disable=broad-except
        print(f"Plotting error: {str(e)}")


def plot_parameter_heatmap(search_results, metric='rmse', save_path=None, figsize=(12, 8)):
    """
    Create heatmap showing metric values across k and lambda parameter space.
    
    Args:
        search_results (dict): Results from search_best_params function
        metric (str): Metric to plot ('rmse', 'mae', or 'r2')
        save_path (str, optional): Path to save the plot
        figsize (tuple): Figure size (width, height)
    """
    try:
        if search_results is None or 'all_results' not in search_results:
            print("Error: No search results to plot")
            return

        results = search_results['all_results']
        if not results:
            print("Error: No results data to plot")
            return

        # Create DataFrame for easier manipulation
        df = pd.DataFrame(results)

        # Create bins for k and lambda
        k_bins = np.linspace(df['k'].min(), df['k'].max(), 10)
        lambda_bins = np.linspace(df['lambda_'].min(), df['lambda_'].max(), 10)

        # Bin the data
        df['k_bin'] = pd.cut(df['k'], k_bins, include_lowest=True)
        df['lambda_bin'] = pd.cut(df['lambda_'], lambda_bins, include_lowest=True)

        # Create pivot table
        metric_col = f'{metric}_mean'
        if metric_col not in df.columns:
            print(f"Error: Metric '{metric}' not found. Available: rmse, mae, r2")
            return

        pivot_table = df.pivot_table(
            values=metric_col,
            index='k_bin',
            columns='lambda_bin',
            aggfunc='mean'
        )

        # Create heatmap
        plt.figure(figsize=figsize)

        # Use appropriate colormap
        cmap = 'viridis_r' if metric == 'rmse' or metric == 'mae' else 'viridis'

        im = plt.imshow(pivot_table.values, cmap=cmap, aspect='auto')

        # Set labels
        plt.title(f'{metric.upper()} Heatmap: Parameter Space Exploration',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Regularization (λ) Bins')
        plt.ylabel('Number of Factors (k) Bins')

        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label(metric.upper())

        # Set tick labels
        plt.xticks(range(len(pivot_table.columns)),
                   [f"{interval.left:.1f}-{interval.right:.1f}"
                    for interval in pivot_table.columns], rotation=45)
        plt.yticks(range(len(pivot_table.index)),
                   [f"{interval.left:.0f}-{interval.right:.0f}"
                    for interval in pivot_table.index])

        plt.tight_layout()

        # Save if requested
        if save_path:
            path = os.path.join("..", "plots", save_path)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {path}")

        plt.show()

    except Exception as e: # pylint: disable=broad-except
        print(f"Heatmap plotting error: {str(e)}")


def plot_convergence_analysis(search_results, save_path=None, figsize=(12, 6)):
    """
    Plot convergence analysis showing how the best score improves over iterations.
    
    Args:
        search_results (dict): Results from search_best_params function
        save_path (str, optional): Path to save the plot
        figsize (tuple): Figure size (width, height)
    """
    try:
        if search_results is None or 'all_results' not in search_results:
            print("Error: No search results to plot")
            return

        results = search_results['all_results']
        if not results:
            print("Error: No results data to plot")
            return

        # Sort by iteration
        sorted_results = sorted(results, key=lambda x: x['iteration'])

        iterations = [r['iteration'] for r in sorted_results]
        rmse_values = [r['rmse_mean'] for r in sorted_results]

        # Calculate running minimum (best so far)
        running_best = []
        current_best = float('inf')
        for rmse in rmse_values:
            if rmse < current_best:
                current_best = rmse
            running_best.append(current_best)

        # Create plot
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left plot: All RMSE values
        ax1.scatter(iterations, rmse_values, alpha=0.6, label='All trials')
        ax1.plot(iterations, running_best, 'r-', linewidth=2, label='Best so far')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('RMSE')
        ax1.set_title('Hyperparameter Search Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right plot: Improvement over iterations
        improvements = []
        for i in range(1, len(running_best)):
            if running_best[i] < running_best[i-1]:
                improvements.append((i+1, running_best[i]))

        if improvements:
            imp_iterations, imp_values = zip(*improvements)
            ax2.plot(imp_iterations, imp_values, 'go-', linewidth=2, markersize=8)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Best RMSE')
            ax2.set_title('Best Score Improvements')
            ax2.grid(True, alpha=0.3)

            # Annotate improvements
            for iter_num, rmse_val in improvements:
                ax2.annotate(f'{rmse_val:.4f}',
                           xy=(iter_num, rmse_val),
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No improvements found',
                    transform=ax2.transAxes, ha='center', va='center')

        plt.tight_layout()

        # Save if requested
        if save_path:
            path = os.path.join("..", "plots", save_path)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to: {path}")

        plt.show()

    except Exception as e: # pylint: disable=broad-except
        print(f"Convergence plotting error: {str(e)}")


def plot_metrics_comparison(search_results, save_path=None, figsize=(15, 5)):
    """
    Compare all three metrics (RMSE, MAE, R²) side by side.
    
    Args:
        search_results (dict): Results from search_best_params function
        save_path (str, optional): Path to save the plot
        figsize (tuple): Figure size (width, height)
    """
    try:
        if search_results is None or 'all_results' not in search_results:
            print("Error: No search results to plot")
            return

        results = search_results['all_results']
        if not results:
            print("Error: No results data to plot")
            return

        # Extract metrics
        rmse_values = [r['rmse_mean'] for r in results]
        mae_values = [r['mae_mean'] for r in results]
        r2_values = [r['r2_mean'] for r in results]

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Metrics Distribution Comparison', fontsize=16, fontweight='bold')

        # RMSE distribution
        axes[0].hist(rmse_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(search_results['best_score'], color='red', linestyle='--',
                       label=f"Best: {search_results['best_score']:.4f}")
        axes[0].set_xlabel('RMSE')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('RMSE Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE distribution
        best_mae_idx = rmse_values.index(min(rmse_values))
        best_mae = mae_values[best_mae_idx]
        axes[1].hist(mae_values, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(best_mae, color='red', linestyle='--',
                       label=f"Best: {best_mae:.4f}")
        axes[1].set_xlabel('MAE')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('MAE Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # R² distribution
        best_r2 = r2_values[best_mae_idx]
        axes[2].hist(r2_values, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[2].axvline(best_r2, color='red', linestyle='--',
                       label=f"Best: {best_r2:.4f}")
        axes[2].set_xlabel('R²')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('R² Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save if requested
        if save_path:
            path = os.path.join("..", "plots", save_path)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison plot saved to: {path}")

        plt.show()

        # Print summary statistics
        print("\nMetrics Summary Statistics:")
        print("="*50)
        print(f"RMSE: μ={np.mean(rmse_values):.4f}, σ={np.std(rmse_values):.4f}, "
              f"min={min(rmse_values):.4f}, max={max(rmse_values):.4f}")
        print(f"MAE:  μ={np.mean(mae_values):.4f}, σ={np.std(mae_values):.4f}, "
              f"min={min(mae_values):.4f}, max={max(mae_values):.4f}")
        print(f"R²:   μ={np.mean(r2_values):.4f}, σ={np.std(r2_values):.4f}, "
              f"min={min(r2_values):.4f}, max={max(r2_values):.4f}")

    except Exception as e: # pylint: disable=broad-except
        print(f"Metrics comparison plotting error: {str(e)}")
