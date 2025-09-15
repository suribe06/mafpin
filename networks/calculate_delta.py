"""
Module for analyzing and visualizing temporal patterns in cascades of user interactions.

A cascade is defined as a sequence of (user_id, timestamp) pairs for a given item.
Timestamps are assumed to be in Unix epoch seconds. Delta (Δ) is the positive time
difference between two user interactions within the same cascade, measured in seconds.

These functions help quantify typical interaction delays, which are useful for
network inference, estimation of transmission rates (α), and other temporal analyses.

Features:
- Plot timelines of user interactions across cascades
- Compute the median time difference (Δ) between interactions

File format:
- Cascades are stored in a text file
- Header block (e.g., user ID labels) followed by cascade data
- Each cascade data line contains comma-separated (user_id, timestamp) pairs
  Example: user1,time1,user2,time2,user3,time3,...

Functions:
    plot_cascades_timestamps(cascade_file, N): 
        Visualize the timelines of the first N cascades

    compute_median_delta(cascade_file): 
        Compute the global median Δ across all cascades (in seconds)

Example:
    delta = compute_median_delta("cascades.txt")
    plot_cascades_timestamps("cascades.txt", N=5)
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

def plot_cascades_timestamps(cascade_file="cascades.txt", n=20):
    """
    Plot the timestamps of user interactions in cascades.
    Args:
        cascade_file (str): Name of the cascades file in the data folder (default: "cascades.txt")
        n (int): Number of cascades to plot (default: 20)
    Returns:
        None
    """
    cascades_path = os.path.join('..', 'data', cascade_file)
    with open(cascades_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Detect when cascade data starts: first line with more than 2 values
    start_idx = 0
    for idx, line in enumerate(lines):
        parts = line.strip().split(",")
        if len(parts) > 2:  # cascade lines have many entries
            start_idx = idx
            break

    cascade_lines = lines[start_idx:start_idx+n]

    plt.figure(figsize=(12, 6))

    for c_idx, line in enumerate(cascade_lines, 1):
        parts = line.strip().split(",")
        pairs = [(int(parts[i]), float(parts[i+1])) for i in range(0, len(parts), 2)]
        # Sort by time
        pairs.sort(key=lambda x: x[1])
        _, times = zip(*pairs)
        dates = [datetime.datetime.fromtimestamp(t) for t in times]

        # Plot this cascade as points connected by lines
        plt.plot(dates, [c_idx]*len(dates), "o-", label=f"Cascade {c_idx}")

    plt.xlabel("Date")
    plt.ylabel("Cascade index")
    plt.title(f"First {n} cascades timelines")
    # plt.legend()
    plt.tight_layout()
    plot_path = os.path.join('..', 'plots', 'cascades_timestamps.png')
    plt.savefig(plot_path)

def compute_median_delta(cascade_file="cascades.txt"):
    """
    Compute the median time difference (delta) between user interactions in cascades.

    Args:
        cascade_file (str): Name of the cascades file in the data folder (default: "cascades.txt")
    Returns:
        float: Median of all positive time differences (delta) between user interactions
    """
    deltas = []

    cascades_path = os.path.join('..', 'data', cascade_file)
    with open(cascades_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Detect when cascade data starts: first line with more than 2 values
    start_idx = 0
    for idx, line in enumerate(lines):
        parts = line.strip().split(",")
        if len(parts) > 2:  # cascade lines have many entries
            start_idx = idx
            break

    # Process cascade lines
    for line in lines[start_idx:]:
        parts = line.strip().split(",")
        if len(parts) < 4:  # need at least 2 (user,time) pairs
            continue
        # Build list of (user, time)
        pairs = [(int(parts[i]), float(parts[i+1])) for i in range(0, len(parts), 2)]
        # Sort by time
        pairs.sort(key=lambda x: x[1])
        times = [t for _, t in pairs]

        # Compute all positive differences
        for i, t_i in enumerate(times):
            for j, t_j in enumerate(times[:i]):
                delta = t_i - t_j
                if delta > 0:
                    deltas.append(delta)

    return np.median(deltas) if deltas else None

# Example usage
cascade_file = "cascades.txt"
plot_cascades_timestamps(cascade_file, 30)
delta_tilde = compute_median_delta(cascade_file)
print("Median positive Δ =", delta_tilde)
