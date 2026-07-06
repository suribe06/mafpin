"""P1-7: core experiment variant CLI flag mapping."""

from __future__ import annotations

import unittest

from recommender.experiment.variants import variant_cli_flags


class VariantFlagTests(unittest.TestCase):
    def test_variant_cli_flags_m2_excludes_social_keys(self) -> None:
        flags = variant_cli_flags("M2")
        self.assertFalse(flags["include_communities"])
        self.assertFalse(flags["social_regularization"])
        self.assertNotIn("social_mode", flags)

    def test_variant_cli_flags_m4a_includes_social_mode(self) -> None:
        flags = variant_cli_flags("M4a")
        self.assertTrue(flags["social_regularization"])
        self.assertEqual(flags["social_mode"], "uniform")
        self.assertEqual(flags["social_normalization"], "mean_weight")

    def test_variant_cli_flags_m4c_robustness_uses_laplacian_norm(self) -> None:
        flags = variant_cli_flags("M4c_robustness")
        self.assertEqual(flags["social_mode"], "boundary_downweight")
        self.assertEqual(flags["social_normalization"], "normalized_laplacian")


if __name__ == "__main__":
    unittest.main()
