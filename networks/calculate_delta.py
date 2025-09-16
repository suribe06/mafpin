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
- Derive α values (centers) from Δ using exponential and Rayleigh models
- Generate log-spaced α grids for inference experiments

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

    alpha_centers_from_delta(delta_seconds): 
        Compute reference α values for exponential and Rayleigh models (in different units)

    log_alpha_grid(alpha0, r, N): 
        Generate a log-spaced α grid around a reference α value

Example:
    delta = compute_median_delta("cascades.txt")
    centers = alpha_centers_from_delta(delta)
    grid = log_alpha_grid(centers["exp_per_day"], r=100, N=30)
    plot_cascades_timestamps("cascades.txt", N=5)
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

SECONDS_PER_DAY = 86400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY

def plot_cascades_timestamps(cascade_file="cascades.txt", n=20):
    """
    Plot the timestamps of user interactions in cascades.

    Args:
        cascade_file (str): Name of the cascades file in the data folder (default: "cascades.txt")
        n (int): Number of cascades to plot (default: 20)

    Returns:
        None (saves a plot to ../plots/cascades_timestamps.png)
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
    plt.tight_layout()
    os.makedirs("../plots", exist_ok=True)
    plot_path = os.path.join('..', 'plots', 'cascades_timestamps.png')
    plt.savefig(plot_path)

def compute_median_delta(cascade_file="cascades.txt"):
    """
    Compute the median time difference (Δ) between user interactions in cascades.

    Args:
        cascade_file (str): Name of the cascades file in the data folder (default: "cascades.txt")

    Returns:
        float: Median of all positive time differences Δ (in seconds)
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

def alpha_centers_from_delta(delta_seconds):
    """
    Compute reference α values from the median delay Δ.

    This function uses the fact that for:
    - Exponential model, the median is m = ln(2)/α  ⇒  α = ln(2)/m
    - Rayleigh model, the median is m = sqrt(2 ln(2))/σ with α = 1/σ²
      ⇒  α = 2 ln(2)/m²

    Args:
        delta_seconds (float): Median Δ in seconds

    Returns:
        dict: α reference values in different units (per second, per day, per year)
              for exponential (s⁻¹) and Rayleigh (s⁻²).
    """
    alpha_exp_s = np.log(2.0) / delta_seconds
    alpha_ray_s = 2.0 * np.log(2.0) / (delta_seconds**2)
    return {
        "exp_per_sec": alpha_exp_s,
        "exp_per_day": alpha_exp_s * SECONDS_PER_DAY,
        "exp_per_year": alpha_exp_s * SECONDS_PER_YEAR,
        "ray_per_sec2": alpha_ray_s,
        "ray_per_day2": alpha_ray_s * (SECONDS_PER_DAY**2),
        "ray_per_year2": alpha_ray_s * (SECONDS_PER_YEAR**2),
    }

def log_alpha_grid(alpha0, r=10.0, N=100):
    """
    Generate a log-spaced α grid centered on a reference value α₀.

    Args:
        alpha0 (float): Reference α value (e.g., from alpha_centers_from_delta)
        r (float): Multiplicative range factor (default: 10). 
                   The grid covers [α₀/r, α₀*r].
        N (int): Number of grid points (default: 100)

    Returns:
        np.ndarray: Log-spaced array of α values
    """
    return np.logspace(np.log10(alpha0/r), np.log10(alpha0*r), N)

# Example usage
if __name__ == "__main__":
    cascade_file = "cascades.txt"
    plot_cascades_timestamps(cascade_file, 30)
    delta_tilde = compute_median_delta(cascade_file)
    print("Median positive Δ (s) =", delta_tilde)

    centers = alpha_centers_from_delta(delta_tilde)
    print("α centers:", centers)

    grid = log_alpha_grid(centers["exp_per_day"], r=10, N=100)
    print("Exponential α grid (per day):", grid)
