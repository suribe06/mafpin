"""
Batch runners for community detection across all inferred networks.
"""

from __future__ import annotations

from pathlib import Path

from config import DatasetPaths, Datasets, Models, Defaults
from networks.network_io import (
    directed_to_undirected,
    load_as_networkx,
    parse_network_filename,
)
from networks.communities.detection import (
    detect_overlapping_communities,
    compute_node_community_membership,
)
from networks.communities.lph import (
    compute_local_pluralistic_homophily,
    compute_lph_paper,
)
from networks.communities.boundary import (
    compute_boundary_indicator,
    save_community_results,
)


def calculate_communities_for_network(
    network_file: str | Path,
    algorithm: str = "demon",
    epsilon: float = Defaults.EPSILON,
    min_community: int = Defaults.MIN_COM,
    output_dir: str | Path | None = None,
    symmetrization: str = "union",
    boundary_percentile: float = 20.0,
) -> bool:
    """
    Detect communities and compute LPH for a single network file.

    The NetInf-inferred directed graph is first converted to undirected using
    *symmetrization* before running community detection and computing h̃v.

    Args:
        network_file:        Path to a NetInf output ``.txt`` file.
        algorithm:           Community detection algorithm (``"demon"`` or ``"aslpaw"``).
        epsilon:             Merging threshold for Demon.
        min_community:       Minimum community size for Demon.
        output_dir:          Target directory for CSV outputs.
        symmetrization:      Method to convert the directed graph to undirected.
            ``"union"`` (default) adds an edge when u→v OR v→u exists;
            ``"intersection"`` requires both directions.
        boundary_percentile: Percentile threshold for the binary boundary flag.

    Returns:
        ``True`` on success, ``False`` otherwise.
    """
    network_file = Path(network_file)
    if not network_file.exists():
        print(f"Error: network file not found: {network_file}")
        return False

    try:
        model_name, network_id = parse_network_filename(network_file.name)
    except ValueError as exc:
        print(f"Error parsing filename: {exc}")
        return False

    print(f"Processing {network_file.name} (model={model_name}, id={network_id})")

    G_directed, user_ids = load_as_networkx(network_file)

    if G_directed.number_of_edges() == 0:
        print(f"  Warning: no edges in {network_file.name}, skipping.")
        return False

    G = directed_to_undirected(G_directed, method=symmetrization)
    print(
        f"  Symmetrization={symmetrization!r}: "
        f"{G_directed.number_of_edges()} directed edges → "
        f"{G.number_of_edges()} undirected edges"
    )

    try:
        communities = detect_overlapping_communities(
            G, algorithm=algorithm, epsilon=epsilon, min_community=min_community
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  Warning: community detection failed: {exc}")
        communities = []

    membership = compute_node_community_membership(user_ids, communities)
    lph = compute_local_pluralistic_homophily(G, membership)
    lph_scores, s_values, delta_values = compute_lph_paper(G, membership)
    boundary_indicator = compute_boundary_indicator(
        lph_scores, percentile=boundary_percentile
    )

    save_community_results(
        user_ids,
        lph,
        lph_scores,
        membership,
        model_name,
        network_id,
        output_dir=output_dir,
        s_values=s_values,
        delta_values=delta_values,
        boundary_indicator=boundary_indicator,
    )
    return True


def calculate_communities_for_all_models(
    algorithm: str = "demon",
    epsilon: float = Defaults.EPSILON,
    min_community: int = Defaults.MIN_COM,
    dataset: str | None = None,
    symmetrization: str = "union",
    boundary_percentile: float = 20.0,
) -> dict[str, int]:
    """
    Detect communities and compute LPH for all inferred networks across all models.

    Args:
        algorithm:           Community detection algorithm.
        epsilon:             Demon merge threshold.
        min_community:       Minimum community size for Demon.
        dataset:             Dataset name.  Defaults to ``Datasets.DEFAULT``.
        symmetrization:      Directed→undirected conversion method.
        boundary_percentile: Percentile threshold for the ``is_boundary`` flag.

    Returns:
        Dict mapping model name → number of successfully processed files.
    """
    dp = DatasetPaths(dataset or Datasets.DEFAULT)
    summary: dict[str, int] = {}
    for model_name in Models.ALL:
        model_dir = dp.NETWORKS / model_name
        if not model_dir.exists():
            print(f"  Skipping {model_name}: directory not found ({model_dir})")
            summary[model_name] = 0
            continue

        network_files = sorted(model_dir.glob("inferred-network-*.txt"))
        if not network_files:
            print(f"  Skipping {model_name}: no network files found")
            summary[model_name] = 0
            continue

        from tqdm import tqdm

        print(f"\n{'='*50}")
        print(f"Model: {model_name.upper()} — {len(network_files)} networks")
        print("=" * 50)

        success_count = 0
        pbar = tqdm(
            network_files,
            desc=f"{model_name[:4].upper()} communities",
            unit="net",
            dynamic_ncols=True,
        )
        for nf in pbar:
            pbar.set_postfix(file=nf.stem[-6:])
            if calculate_communities_for_network(
                nf,
                algorithm=algorithm,
                epsilon=epsilon,
                min_community=min_community,
                output_dir=dp.COMMUNITIES / model_name,
                symmetrization=symmetrization,
                boundary_percentile=boundary_percentile,
            ):
                success_count += 1
        pbar.close()

        summary[model_name] = success_count
        print(f"Completed: {success_count}/{len(network_files)}")

    return summary
