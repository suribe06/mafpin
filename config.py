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
    NETINF_BIN = ROOT / "networks" / "netinf"


class DatasetPaths:
    """Per-dataset output paths, scoped under ``data/<dataset>/``.

    All generated artefacts (cascades, networks, centrality metrics, etc.) are
    stored inside a dataset-specific subdirectory so that results from
    different datasets never overwrite each other.

    Example::

        dp = DatasetPaths("movielens")
        dp.CASCADES    # data/movielens/cascades.txt
        dp.NETWORKS    # data/movielens/inferred_networks/
        dp.PLOTS       # plots/movielens/
    """

    def __init__(self, dataset: str) -> None:
        base = Paths.DATA / dataset
        self.BASE = base
        self.CASCADES = base / "cascades.txt"
        self.CASCADE_USER_STATS = base / "cascade_user_stats.csv"
        self.NETWORKS = base / "inferred_networks"
        self.CENTRALITY = base / "centrality_metrics"
        self.COMMUNITIES = base / "communities"
        self.SHAP_MATRICES = base / "shap_matrices"
        self.PLOTS = Paths.PLOTS / dataset
        self.BASELINE_RESULTS = base / "baseline_search_results.json"
        self.ENHANCED_RESULTS = base / "enhanced_search_results.json"
        self.SHAP_RESULTS = base / "shap_results.json"


class Datasets:
    """Paths and format configurations for supported rating datasets.

    Each entry in :attr:`CONFIG` describes how to read the raw file for a
    given dataset.  Column indices (``col_user``, ``col_item``,
    ``col_rating``, ``col_time``) are zero-based positions in the raw file
    and are used to extract the four fields needed by the pipeline.
    """

    ROOT = Path(__file__).parent / "datasets"
    DEFAULT = "movielens"
    ALL = ["movielens", "ciao", "epinions"]

    CONFIG: "dict[str, dict]" = {
        "movielens": {
            "file": "ratings_small.csv",
            "sep": ",",
            "header": 0,  # first row is a header
            "col_user": 0,
            "col_item": 1,
            "col_rating": 2,
            "col_time": 3,
        },
        "ciao": {
            "file": "rating_with_timestamp.txt",
            "sep": r"\s+",
            "header": None,  # no header row
            "col_user": 0,
            "col_item": 1,
            "col_rating": 3,  # columns: user, product, category, rating, helpfulness, time
            "col_time": 5,
        },
        "epinions": {
            "file": "rating_with_timestamp.txt",
            "sep": r"\s+",
            "header": None,  # no header row
            "col_user": 0,
            "col_item": 1,
            "col_rating": 3,  # columns: user, product, category, rating, helpfulness, time
            "col_time": 5,
        },
    }


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
    MAX_ITER = 5000  # fallback k (edges) when k_avg_degree is not used
    K_AVG_DEGREE = 2  # k = avg_degree × N  (paper: real nets ≈ 1–4 out-edges per node)

    # Community detection
    EPSILON = 0.25  # Demon merge threshold (lower → more communities)
    MIN_COM = 3  # minimum community size kept by Demon

    # Matrix factorization
    K = 20  # number of latent factors
    LAMBDA_REG = 1.0  # regularisation coefficient

    # CMF side-information weights (enhanced model only)
    W_MAIN = 1.0  # weight for main rating-matrix reconstruction loss
    W_USER = 0.1  # weight for user side-information reconstruction loss


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

    ``STRATEGY`` controls the split method:
    * ``"random"``   — shuffled split (sklearn ``train_test_split``); fast but
      may place earlier interactions in the test set.
    * ``"temporal"`` — last ``TEST_SIZE`` fraction of each user's interactions
      (by timestamp) are held out; guarantees test ratings are always *later*
      than training ratings, simulating realistic deployment.
    """

    TEST_SIZE = 0.2  # fraction of ratings held out for testing
    RANDOM_STATE = 42  # seed for train_test_split (random strategy only)
    STRATEGY = "temporal"  # "random" or "temporal"


# ---------------------------------------------------------------------------
# MLflow tracking
# ---------------------------------------------------------------------------


class MLflow:
    """MLflow experiment tracking configuration.

    By default MLflow stores runs under ``mlruns/`` in the project root
    (file-based, no server required).  Run ``mlflow ui`` from the project
    root to browse results at http://127.0.0.1:5000.
    """

    EXPERIMENT_NAME = "mafpin"
    TRACKING_URI = str(ROOT / "mlruns")


# ---------------------------------------------------------------------------
# Side-user feature selection
# ---------------------------------------------------------------------------


class SideUserFeatures:
    """Toggle individual side-user features for the enhanced CMF model.

    Set a feature to ``False`` to exclude it from the user-attribute matrix
    passed to :class:`cmfrec.CMF`.  The filter is applied at the end of
    :func:`recommender.enhanced.load_network_features`, so disabling a feature
    here has no effect on which CSV files are read — only on which columns are
    kept in the returned DataFrame.

    Feature groups
    --------------
    Centrality metrics  — loaded from ``centrality_metrics_<model>_<idx>.csv``
    Community features  — loaded from ``communities_<model>_<idx>.csv``
    Cascade statistics  — loaded from ``cascade_user_stats.csv``

    ``community_binary`` is a special key that controls the dynamically-generated
    binary membership columns (``community_<id>``).  Setting it to ``False``
    drops all such columns regardless of how many communities were encoded.
    """

    FEATURES: "dict[str, bool]" = {
        # --- Centrality metrics ---
        "degree": True,
        "in_degree": False,
        "out_degree": False,
        "betweenness": True,
        "closeness": True,
        "eigenvector": True,
        "pagerank": True,
        "clustering": True,
        "eccentricity": True,
        "hub_score": False,
        "auth_score": False,
        "pagerank_lph": True,
        # --- Community features ---
        "local_pluralistic_hom": True,
        "num_communities": False,
        "lph_score": True,
        "s_v": True,          # s(v): neighborhood alignment count (Eq. 3, Barraza 2025)
        "delta_v": True,      # δv: local dissimilarity (Eq. 4, Barraza 2025)
        "is_boundary": True,  # binary boundary flag (h̃v ≤ p-th percentile)
        "community_binary": False,  # binary community_<id> membership columns
        # --- Cascade statistics ---
        "mean_cascade_position": False,
        "min_cascade_position": False,
        "cascade_breadth": False,
    }
