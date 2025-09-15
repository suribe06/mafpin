"""
This module provides functions for calculating and visualizing various centrality metrics
in network graphs using SNAP-py library.

The module includes functions to calculate and plot:
- Degree centrality
- Betweenness centrality
- Closeness centrality
- Eigenvector centrality
- PageRank
- Clustering coefficient
- Eccentricity

Each function takes a SNAP graph object as input and returns a list of metric values
for each node in the graph. Optional parameters allow for visualization of the
metric distributions using histograms with kernel density estimation.
The module uses matplotlib and seaborn for visualization, and numpy for basic
statistical calculations. All functions include error handling and logging of
basic statistics like average metric values.
Dependencies:
    - os
    - matplotlib.pyplot
    - seaborn
    - numpy
    - snap (Stanford Network Analysis Platform)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_metric_distribution(data, metric_name, model_name, network_id, output_dir):
    """
    Plot the distribution of a centrality metric.
    
    Args:
        data (list): Metric values
        metric_name (str): Name of the metric
        model_name (str): Model name
        network_id (str): Network identifier
        output_dir (str): Output directory for plots
    """
    if not data or len(data) == 0:
        print(f"Warning: No data for {metric_name} metric")
        return

    plt.figure(figsize=(10, 6))
    plt.clf()

    try:
        # Filter out zero or negative values for log scale
        filtered_data = [x for x in data if x > 0]
        if len(filtered_data) == 0:
            print(f"Warning: No positive values for {metric_name} metric")
            return

        sns.histplot(filtered_data, kde=True, color='darkblue', alpha=0.7)
        plt.yscale('log')
        plt.xlabel(metric_name)
        plt.title(f'{metric_name} Distribution - {model_name} Model (Network {network_id})')
        plt.legend(labels=['KDE', f'{metric_name} Distribution'])
        plt.grid(True, alpha=0.3)

        plot_file = os.path.join(
            output_dir,
            f'{metric_name}_distribution_{model_name}_{network_id}.png'
        )
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e: # pylint: disable=broad-except
        print(f"Error plotting {metric_name}: {str(e)}")
        plt.close()

def calculate_degree_snap(G, plot_enabled=False, model_name="", network_id="", output_dir=""):
    """
    Calculate degree distribution using SNAP-py
    
    Args:
        G: SNAP graph object
        plot_enabled (bool): Whether to create plots
        model_name (str): Model name for plot titles
        network_id (str): Network ID for plot titles
        output_dir (str): Output directory for plots
    
    Returns:
        list: List of node degrees
    """
    try:
        # Get degree distribution using SNAP
        degree_counts = []
        degrees = []

        for node in G.Nodes():
            node_id = node.GetId()
            node_degree = G.GetDegreeCentr(node_id)
            degrees.append(node_degree)
            degree_counts.append(node_degree)

        avg_degree = sum(degrees) / G.GetNodes() if G.GetNodes() > 0 else 0
        print(f"Average Degree = {avg_degree:.4f}")

        if plot_enabled and len(degrees) > 0:
            plot_metric_distribution(degrees, "Degree", model_name, network_id, output_dir)

        return degrees

    except Exception as e: # pylint: disable=broad-except
        print(f"Error calculating degree distribution: {str(e)}")
        return []


def calculate_betweenness_snap(G, plot_enabled=False, model_name="", network_id="", output_dir=""):
    """
    Calculate betweenness centrality using SNAP-py
    
    Args:
        G: SNAP graph object
        plot_enabled (bool): Whether to create plots
        model_name (str): Model name for plot titles
        network_id (str): Network ID for plot titles
        output_dir (str): Output directory for plots
    
    Returns:
        list: List of betweenness centrality values
    """
    n = G.GetNodes()
    m = G.GetEdges()
    print(f"Calculating Betweenness Centrality for graph with {n} nodes and {m} edges...")

    try:
        # Get betweenness centrality using SNAP
        nodes_betweenness, _ = G.GetBetweennessCentr(1.0)
        betweenness = [
            nodes_betweenness[node.GetId()] if node.GetId() in nodes_betweenness else 0.0
            for node in G.Nodes()
        ]
        avg_betweenness = np.mean(betweenness) if betweenness else 0
        print(f"Average Betweenness = {avg_betweenness:.6f}")

        if plot_enabled and len(betweenness) > 0:
            plot_metric_distribution(betweenness, "Betweenness Centrality", model_name, network_id, output_dir)

        return betweenness

    except Exception as e: # pylint: disable=broad-except
        print(f"Error calculating betweenness centrality: {str(e)}")
        return [0.0] *n

def calculate_closeness_snap(G, plot_enabled=False, model_name="", network_id="", output_dir=""):
    """
    Calculate closeness centrality using SNAP-py
    
    Args:
        G: SNAP graph object
        plot_enabled (bool): Whether to create plots
        model_name (str): Model name for plot titles
        network_id (str): Network ID for plot titles
        output_dir (str): Output directory for plots
    
    Returns:
        list: List of closeness centrality values
    """
    n = G.GetNodes()
    m = G.GetEdges()
    print(f"Calculating Closeness Centrality for graph with {n} nodes and {m} edges...")

    try:
        # Get closeness centrality using SNAP
        closeness = []
        for node in G.Nodes():
            node_id = node.GetId()
            try:
                cc = G.GetClosenessCentr(node_id)
                closeness.append(cc)
            except (TypeError, AttributeError):
                closeness.append(0.0)

        avg_closeness = np.mean(closeness) if closeness else 0
        print(f"Average Closeness = {avg_closeness:.6f}")

        if plot_enabled and len(closeness) > 0:
            plot_metric_distribution(closeness, "Closeness Centrality", model_name, network_id, output_dir)

        return closeness

    except Exception as e: # pylint: disable=broad-except
        print(f"Error calculating closeness centrality: {str(e)}")
        return [0.0] * n

def calculate_eigenvector_snap(G, plot_enabled=False, model_name="", network_id="", output_dir=""):
    """
    Calculate eigenvector centrality using SNAP-py
    
    Args:
        G: SNAP graph object
        plot_enabled (bool): Whether to create plots
        model_name (str): Model name for plot titles
        network_id (str): Network ID for plot titles
        output_dir (str): Output directory for plots
    
    Returns:
        list: List of eigenvector centrality values
    """
    n = G.GetNodes()
    m = G.GetEdges()
    print(f"Calculating Eigenvector Centrality for graph with {n} nodes and {m} edges...")

    try:
        eigenvector_dict = G.GetEigenVectorCentr()
        eigenvector = []
        for node in G.Nodes():
            node_id = node.GetId()
            if node_id in eigenvector_dict:
                eigenvector.append(eigenvector_dict[node_id])
            else:
                eigenvector.append(0.0)

        avg_eigenvector = np.mean(eigenvector) if eigenvector else 0
        print(f"Average Eigenvector = {avg_eigenvector:.6f}")

        if plot_enabled and len(eigenvector) > 0:
            plot_metric_distribution(eigenvector, "Eigenvector Centrality", model_name, network_id, output_dir)

        return eigenvector

    except Exception as e: # pylint: disable=broad-except
        print(f"Error calculating eigenvector centrality: {str(e)}")
        return [0.0] * n

def calculate_pagerank_snap(G, plot_enabled=False, model_name="", network_id="", output_dir=""):
    """
    Calculate PageRank using SNAP-py
    
    Args:
        G: SNAP graph object
        plot_enabled (bool): Whether to create plots
        model_name (str): Model name for plot titles
        network_id (str): Network ID for plot titles
        output_dir (str): Output directory for plots
    
    Returns:
        list: List of PageRank values
    """
    n = G.GetNodes()
    m = G.GetEdges()
    print(f"Calculating PageRank for graph with {n} nodes and {m} edges...")

    try:
        pagerank_dict = G.GetPageRank()
        pagerank = []
        for node in G.Nodes():
            node_id = node.GetId()
            if node_id in pagerank_dict:
                pagerank.append(pagerank_dict[node_id])
            else:
                pagerank.append(0.0)

        avg_pagerank = np.mean(pagerank) if pagerank else 0
        print(f"Average PageRank = {avg_pagerank:.6f}")

        if plot_enabled and len(pagerank) > 0:
            plot_metric_distribution(pagerank, "PageRank", model_name, network_id, output_dir)

        return pagerank

    except Exception as e: # pylint: disable=broad-except
        print(f"Error calculating PageRank: {str(e)}")
        return [0.0] * n

def calculate_clustering_snap(G, plot_enabled=False, model_name="", network_id="", output_dir=""):
    """
    Calculate clustering coefficient using SNAP-py
    
    Args:
        G: SNAP graph object
        plot_enabled (bool): Whether to create plots
        model_name (str): Model name for plot titles
        network_id (str): Network ID for plot titles
        output_dir (str): Output directory for plots
    
    Returns:
        list: List of clustering coefficient values
    """
    n = G.GetNodes()
    m = G.GetEdges()
    print(f"Calculating Clustering Coefficient for graph with {n} nodes and {m} edges...")

    try:
        clustering = []
        for node in G.Nodes():
            node_id = node.GetId()
            try:
                cc = G.GetNodeClustCf(node_id)
                clustering.append(cc)
            except (TypeError, AttributeError):
                clustering.append(0.0)

        avg_clustering = np.mean(clustering) if clustering else 0
        print(f"Average Clustering Coefficient = {avg_clustering:.6f}")

        if plot_enabled and len(clustering) > 0:
            plot_metric_distribution(clustering, "Clustering Coefficient", model_name, network_id, output_dir)

        return clustering

    except Exception as e: # pylint: disable=broad-except
        print(f"Error calculating clustering coefficient: {str(e)}")
        return [0.0] * n

def calculate_eccentricity_snap(G, plot_enabled=False, model_name="", network_id="", output_dir=""):
    """
    Calculate eccentricity using SNAP-py
    
    Args:
        G: SNAP graph object
        plot_enabled (bool): Whether to create plots
        model_name (str): Model name for plot titles
        network_id (str): Network ID for plot titles
        output_dir (str): Output directory for plots
    
    Returns:
        list: List of eccentricity values
    """
    n = G.GetNodes()
    m = G.GetEdges()
    print(f"Calculating Eccentricity for graph with {n} nodes and {m} edges...")

    try:
        eccentricity = []
        for node in G.Nodes():
            node_id = node.GetId()
            try:
                ecc = G.GetNodeEcc(node_id)
                eccentricity.append(ecc)
            except (TypeError, AttributeError):
                eccentricity.append(0)

        avg_eccentricity = np.mean(eccentricity) if eccentricity else 0
        print(f"Average Eccentricity = {avg_eccentricity:.6f}")

        if plot_enabled and len(eccentricity) > 0:
            plot_metric_distribution(eccentricity, "Eccentricity", model_name, network_id, output_dir)

        return eccentricity

    except Exception as e: # pylint: disable=broad-except
        print(f"Error calculating eccentricity: {str(e)}")
        return [0] * n
