"""P2-3: pipeline CLI contract for core experiment steps."""

from __future__ import annotations

import unittest

from pipeline._cli import _build_parser


class PipelineCliTests(unittest.TestCase):
    def test_parser_accepts_phase2_step_chain(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--steps",
                "import_manifest",
                "canonical_baseline",
                "network_selection",
                "final_eval",
                "--dataset",
                "movielens",
            ]
        )
        self.assertEqual(
            args.steps,
            ["import_manifest", "canonical_baseline", "network_selection", "final_eval"],
        )
        self.assertEqual(args.dataset, "movielens")

    def test_parser_accepts_single_model_variant_for_final_eval(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            ["--steps", "final_eval", "--model-variant", "M4c", "--dataset", "ciao"]
        )
        self.assertEqual(args.model_variant, "M4c")
        self.assertEqual(args.dataset, "ciao")

    def test_parser_social_regularization_flags(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--steps",
                "recommend",
                "--social-regularization",
                "--social-mode",
                "boundary_downweight",
                "--social-normalization",
                "mean_weight",
            ]
        )
        self.assertTrue(args.social_regularization)
        self.assertEqual(args.social_mode, "boundary_downweight")
        self.assertEqual(args.social_normalization, "mean_weight")


if __name__ == "__main__":
    unittest.main()
