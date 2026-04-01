"""
Central configuration for MAFPIN.

All file-system paths and default parameter values are defined here so that
every other module can import them instead of hard-coding relative paths.
Running scripts from any working directory is therefore safe.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root and data paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent  # project root, regardless of cwd


class Paths:
    """Absolute paths to every data and output directory used by the project."""

    DATA = ROOT / "data"
    PLOTS = ROOT / "plots"
    CASCADES = DATA / "cascades.txt"
    NETWORKS = DATA / "inferred_networks"
    CENTRALITY = DATA / "centrality_metrics"
    COMMUNITIES = DATA / "communities"
    NETINF_BIN = ROOT / "networks" / "netinf"


# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------


class Models:
    """Canonical names and short codes for the three diffusion models."""

    ALL = ["exponential", "powerlaw", "rayleigh"]
    SHORT = {"exponential": "expo", "powerlaw": "power", "rayleigh": "ray"}
    FROM_SHORT = {v: k for k, v in SHORT.items()}


# ---------------------------------------------------------------------------
# Default algorithm parameters
# ---------------------------------------------------------------------------


class Defaults:
    """Default values for every tunable parameter in the pipeline."""

    # Network inference
    N_ALPHAS = 100  # number of alpha values in the log-spaced grid
    RANGE_R = 100.0  # multiplicative range factor: grid spans [center/r, center*r]
    MAX_ITER = 20000  # maximum NetInf iterations per network

    # Community detection
    EPSILON = 0.25  # Demon merge threshold (lower → more communities)
    MIN_COM = 3  # minimum community size kept by Demon

    # Matrix factorization
    K = 20  # number of latent factors
    LAMBDA_REG = 1.0  # regularisation coefficient


# ---------------------------------------------------------------------------
# Global train / test split
# ---------------------------------------------------------------------------


class Split:
    """Parameters for the single global train/test split applied at the start
    of the pipeline.

    Using one fixed split ensures that:
    * Cascade generation (NetInf input) sees only training interactions.
    * CMF training and feature scaling never touch held-out ratings.
    * Results are reproducible across pipeline re-runs.
    """

    TEST_SIZE = 0.2  # fraction of ratings held out for testing
    RANDOM_STATE = 42  # seed for train_test_split — change to re-randomise
