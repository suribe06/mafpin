"""
A module for calculating and analyzing various centrality metrics in network graphs using SNAP-py.
This module provides functionality to:
1. Load inferred networks from files
2. Calculate multiple centrality metrics (degree, betweenness, closeness, eigenvector, 
    pagerank, clustering, and eccentricity)
3. Save results to CSV files
4. Process multiple networks across different models (exponential, powerlaw, rayleigh)
The module uses SNAP-py for efficient graph processing and includes capabilities for:
- Creating necessary directory structures
- Loading and converting network data to SNAP graph format
- Calculating comprehensive centrality metrics
- Saving results in structured CSV format
- Batch processing of multiple networks
- Optional visualization of results
Dependencies:
     - os
     - glob
     - pandas
     - snap
     - centrality_metrics (local module)
Note:
     The module expects a specific directory structure and file naming convention
     for the inferred networks. Network files should be stored in:
     '../data/inferred_networks/<model_name>/'
"""

import os
import glob
import pandas as pd
from snap import snap

from centrality_metrics import (
    calculate_degree_snap,
    calculate_betweenness_snap,
    calculate_closeness_snap,
    calculate_eigenvector_snap,
    calculate_pagerank_snap,
    calculate_clustering_snap,
    calculate_eccentricity_snap,
)

def create_output_directories(base_path, model_name):
    """
    Create necessary output directories for centrality metrics.
    
    Args:
        base_path (str): Base path for centrality metrics
        model_name (str): Name of the model (exponential, powerlaw, rayleigh)
    
    Returns:
        str: Path to the model-specific directory
    """
    model_dir = os.path.join(base_path, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def calculate_centrality_metrics_snap(
        G, plot_enabled=False, model_name="", network_id="", output_dir=""
    ):
    """
    Calculate various centrality metrics using SNAP-py.
    
    Args:
        G: SNAP graph object
        plot_enabled (bool): Whether to create plots
        model_name (str): Model name for plot titles
        network_id (str): Network ID for plot titles
        output_dir (str): Output directory for plots
    
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    n = G.GetNodes()
    m = G.GetEdges()

    print(f"Calculating metrics for graph with {n} nodes and {m} edges")

    metrics = {}

    try:
        metrics['degrees'] = calculate_degree_snap(G, plot_enabled, model_name, network_id, output_dir)
        metrics['betweenness'] = calculate_betweenness_snap(G, plot_enabled, model_name, network_id, output_dir)
        metrics['closeness'] = calculate_closeness_snap(G, plot_enabled, model_name, network_id, output_dir)
        metrics['eigenvector'] = calculate_eigenvector_snap(G, plot_enabled, model_name, network_id, output_dir)
        metrics['pagerank'] = calculate_pagerank_snap(G, plot_enabled, model_name, network_id, output_dir)
        metrics['clustering'] = calculate_clustering_snap(G, plot_enabled, model_name, network_id, output_dir)
        metrics['eccentricity'] = calculate_eccentricity_snap(G, plot_enabled, model_name, network_id, output_dir)
        return metrics

    except Exception as e: # pylint: disable=broad-except
        print(f"Error in centrality metrics calculation: {str(e)}")
        return {
            'degrees': [0] * n,
            'betweenness': [0.0] * n,
            'closeness': [0.0] * n,
            'eigenvector': [0.0] * n,
            'pagerank': [0.0] * n,
            'clustering': [0.0] * n,
            'eccentricity': [0] * n,
        }

def save_centrality_results(user_ids, metrics, model_name, network_id, output_dir):
    """
    Save centrality metrics to CSV file.
    
    Args:
        user_ids (list): List of user IDs
        metrics (dict): Dictionary of calculated metrics
        model_name (str): Model name
        network_id (str): Network identifier
        output_dir (str): Output directory
    """
    try:
        # Prepare data
        data = {
            'UserId': user_ids,
            'degree': metrics.get('degrees', []),
            'betweenness': metrics.get('betweenness', []),
            'closeness': metrics.get('closeness', []),
            'eigenvector': metrics.get('eigenvector', []),
            'pagerank': metrics.get('pagerank', []),
            'clustering': metrics.get('clustering', []),
            'eccentricity': metrics.get('eccentricity', []),
        }

        # Create DataFrame
        df = pd.DataFrame(data)

        # Save to CSV
        output_file = os.path.join(output_dir, f'centrality_metrics_{model_name}_{network_id}.csv')
        df.to_csv(output_file, index=False)

        print(f"Centrality metrics saved to: {output_file}")

    except Exception as e: # pylint: disable=broad-except
        print(f"Error saving results: {str(e)}")

def load_inferred_network_snap(network_file):
    """
    Load an inferred network file and create SNAP graph.
    
    Args:
        network_file (str): Path to the network file
        
    Returns:
        tuple: (SNAP graph, list of user IDs)
    """
    try:
        # Load the network file
        if not os.path.exists(network_file):
            print(f"Error: Network file '{network_file}' not found.")
            return None, []

        with open(network_file, 'r', encoding='utf-8') as f:
            lines = f.read().strip().splitlines()

        nodes = []
        edges = []
        for line in lines:
            if not line.strip():
                continue
            i, j = map(int, line.split(","))
            if i == j:
                nodes.append(i)
            else:
                edges.append((i, j))

        mapper = {old_id: new_id for new_id, old_id in enumerate(sorted(set(nodes)), start=1)}

        G = snap.TUNGraph.New()
        user_ids = list(mapper.values())
        for u in user_ids:
            G.AddNode(u)
        for i, j in edges:
            if i in mapper and j in mapper:
                G.AddEdge(mapper[i], mapper[j])

        print(f"Loaded network: {G.GetNodes()} nodes, {G.GetEdges()} edges")
        return G, user_ids

    except Exception as e: # pylint: disable=broad-except
        print(f"Error loading network file '{network_file}': {str(e)}")
        return None, []

def calculate_centrality_for_network(network_file, plot_enabled=False):
    """
    Calculate centrality metrics for a specific inferred network.
    
    Args:
        network_file (str): Path to the inferred network file
        plot_enabled (bool): Whether to generate plots
        
    Returns:
        bool: Success status
    """
    try:
        # Extract model and network info from filename
        filename = os.path.basename(network_file)
        parts = filename.replace('.txt', '').split('-')
        if len(parts) < 4:
            print(f"Error: Invalid network filename format: {filename}")
            return False
        model_short = parts[2]  # expo, power, ray
        network_id = parts[3]   # network number
        # Map model names
        model_mapping = {'expo': 'exponential', 'power': 'powerlaw', 'ray': 'rayleigh'}
        model_name = model_mapping.get(model_short, model_short)

        print(f"\n{'='*60}")
        print(f"Processing {model_name} model - Network {network_id}")
        print(f"File: {network_file}")
        print(f"{'='*60}")

        # Create output directory
        base_output_dir = os.path.join('..', 'data', 'centrality_metrics')
        model_output_dir = create_output_directories(base_output_dir, model_name)

        # Load the network
        G, user_ids = load_inferred_network_snap(network_file)
        if G is None or G.GetNodes() == 0:
            print("Error: Could not load network or network is empty")
            return False

        # Calculate centrality metrics
        metrics = calculate_centrality_metrics_snap(
            G, plot_enabled, model_name, network_id, model_output_dir
        )
        # Save results
        save_centrality_results(user_ids, metrics, model_name, network_id, model_output_dir)
        print(f"Successfully processed {model_name} network {network_id}")
        return True

    except Exception as e: # pylint: disable=broad-except
        print(f"Error processing network {network_file}: {str(e)}")
        return False

def calculate_centrality_for_all_models(plot_enabled=False):
    """
    Calculate centrality metrics for all inferred networks (all models).
    
    Args:
        plot_enabled (bool): Whether to generate plots
        
    Returns:
        dict: Results summary for each model
    """
    try:
        # Base path for inferred networks
        base_network_path = os.path.join('..', 'data', 'inferred_networks')

        if not os.path.exists(base_network_path):
            print(f"Error: Inferred networks directory not found: {base_network_path}")
            return {}

        results = {}
        models = ['exponential', 'powerlaw', 'rayleigh']

        for model in models:
            model_path = os.path.join(base_network_path, model)

            if not os.path.exists(model_path):
                print(f"Warning: Model directory not found: {model_path}")
                results[model] = {'processed': 0, 'failed': 0}
                continue

            # Find all network files for this model
            pattern = os.path.join(model_path, '*.txt')
            network_files = glob.glob(pattern)

            # Filter out CSV files
            network_files = [f for f in network_files if not f.endswith('.csv')]

            if not network_files:
                print(f"Warning: No network files found for {model} model")
                results[model] = {'processed': 0, 'failed': 0}
                continue

            print(f"\n{'='*80}")
            print(f"Processing {model.upper()} model - {len(network_files)} networks found")
            print(f"{'='*80}")

            processed = 0
            failed = 0

            for network_file in sorted(network_files):
                try:
                    success = calculate_centrality_for_network(network_file, plot_enabled)
                    if success:
                        processed += 1
                    else:
                        failed += 1
                except Exception as e: # pylint: disable=broad-except
                    print(f"Error processing {network_file}: {str(e)}")
                    failed += 1

            results[model] = {'processed': processed, 'failed': failed}
            print(f"\n{model.capitalize()} model summary: {processed} processed, {failed} failed")

        # Print overall summary
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}")
        total_processed = sum(r['processed'] for r in results.values())
        total_failed = sum(r['failed'] for r in results.values())

        for model, result in results.items():
            print(f"{model.capitalize()}: {result['processed']} processed, {result['failed']} failed")

        print(f"\nTotal: {total_processed} processed, {total_failed} failed")

        return results

    except Exception as e: # pylint: disable=broad-except
        print(f"Error in batch processing: {str(e)}")
        return {}
