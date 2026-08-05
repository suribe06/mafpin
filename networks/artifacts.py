"""Network artefact path locator — single seam for filename conventions."""

from __future__ import annotations

from pathlib import Path

from config import DatasetPaths, Models


class NetworkArtifacts:
    """Locate inferred-network / centrality / community files for a dataset.

    Callers should not reconstruct ``inferred-network-{short}-{idx}`` strings.
    Pass an optional ``paths`` (e.g. ``ColdStartPaths``) to root elsewhere.
    """

    def __init__(
        self,
        dataset: str,
        paths: DatasetPaths | None = None,
    ) -> None:
        self.dataset = dataset
        self.dp = paths or DatasetPaths(dataset)

    def network_txt(self, model_name: str, network_index: int) -> Path:
        short = Models.SHORT[model_name]
        return (
            self.dp.NETWORKS
            / model_name
            / f"inferred-network-{short}-{network_index:03d}.txt"
        )

    def centrality_csv(self, model_name: str, network_index: int) -> Path:
        return (
            self.dp.CENTRALITY
            / model_name
            / f"centrality_metrics_{model_name}_{network_index:03d}.csv"
        )

    def communities_csv(self, model_name: str, network_index: int) -> Path:
        return (
            self.dp.COMMUNITIES
            / model_name
            / f"communities_{model_name}_{network_index:03d}.csv"
        )

    def inferred_edges_csv(self, model_name: str) -> Path:
        short = Models.SHORT[model_name]
        return self.dp.NETWORKS / model_name / f"inferred_edges_{short}.csv"

    def list_centrality_indices(self, model_name: str) -> list[int]:
        pattern = f"centrality_metrics_{model_name}_*.csv"
        return sorted(
            {
                idx
                for idx in (
                    _index_from_underscore(p)
                    for p in (self.dp.CENTRALITY / model_name).glob(pattern)
                )
                if idx is not None
            }
        )

    def list_network_indices(self, model_name: str) -> list[int]:
        short = Models.SHORT[model_name]
        pattern = f"inferred-network-{short}-*.txt"
        return sorted(
            {
                idx
                for idx in (
                    _index_from_dash(p)
                    for p in (self.dp.NETWORKS / model_name).glob(pattern)
                )
                if idx is not None
            }
        )

    def list_community_indices(self, model_name: str) -> list[int]:
        pattern = f"communities_{model_name}_*.csv"
        return sorted(
            {
                idx
                for idx in (
                    _index_from_underscore(p)
                    for p in (self.dp.COMMUNITIES / model_name).glob(pattern)
                )
                if idx is not None
            }
        )

    def list_complete_indices(self, model_name: str) -> list[int]:
        """Indices that have network + centrality + communities artefacts."""
        return sorted(
            set(self.list_centrality_indices(model_name))
            & set(self.list_network_indices(model_name))
            & set(self.list_community_indices(model_name))
        )


def _index_from_underscore(path: Path) -> int | None:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except (AttributeError, TypeError, ValueError):
        return None


def _index_from_dash(path: Path) -> int | None:
    try:
        return int(path.stem.rsplit("-", 1)[-1])
    except (AttributeError, TypeError, ValueError):
        return None
