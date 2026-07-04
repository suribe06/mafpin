"""Self-check for experiment manifest import and network selection helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from recommender.experiment.manifest import build_manifest_from_logs
from recommender.experiment.network_selection import _pick_from_network_best


def test_pick_from_network_best() -> None:
    network_best = {
        "exponential": {"alpha": 0.15, "cv_rmse": 0.89},
        "powerlaw": {"alpha": 2.1, "cv_rmse": 0.91},
        "rayleigh": {"alpha": 0.02, "cv_rmse": 0.95},
    }
    picked = _pick_from_network_best(network_best)
    assert picked is not None
    assert picked["diffusion_model"] == "exponential"
    assert picked["cv_rmse"] == 0.89


def test_build_manifest_from_logs() -> None:
    log_text = """
[RECOMMEND] Done.
Enhanced CMF best hyperparameters:
{
  "k": 9,
  "lambda_reg": 0.5,
  "w_main": 1.0,
  "w_user": 0.1
}
Exponential — Best α=1.5119e-01  RMSE=0.889019  improvement=+0.30%
"""
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        (log_dir / "m3_recommend.log").write_text(log_text, encoding="utf-8")
        manifest = build_manifest_from_logs(
            "movielens",
            log_dir=log_dir,
            variant_ids=["M3"],
        )
        assert "M3" in manifest["variants"]
        assert manifest["variants"]["M3"]["hyperparameters"]["k"] == 9
        assert "exponential" in manifest["variants"]["M3"]["network_best"]


if __name__ == "__main__":
    test_pick_from_network_best()
    test_build_manifest_from_logs()
    print("ok")
