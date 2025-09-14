#!/usr/bin/env python3
"""
Network Inference Runner Script

This script provides a command-line interface to run network inference
with custom parameters.

Usage examples:
- python run_inference.py --help
- python run_inference.py --model 0 --N 50 --alpha-0 1e-6 --alpha-f 1e-2
- python run_inference.py --all-models --N 20
- python run_inference.py --cascades custom_cascades.txt --model 1 --max-iter 100000
"""

import argparse
import sys
from infer_networks import infer_networks, infer_networks_all_models

def main():
    """Main function to parse arguments and run inference."""

    parser = argparse.ArgumentParser(
        description='Infer networks from cascade data using netinf executable',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--cascades', 
        type=str,
        default='cascades.txt',
        help='Name of cascades file in the data folder'
    )

    parser.add_argument(
        '--alpha-0', 
        type=float,
        default=1e-7,
        help='Initial alpha value (transmission rate)'
    )

    parser.add_argument(
        '--alpha-f', 
        type=float,
        default=1e-3,
        help='Final alpha value (transmission rate)'
    )

    parser.add_argument(
        '--N', 
        type=int,
        default=100,
        help='Number of different alpha values to generate'
    )

    parser.add_argument(
        '--model', 
        type=int,
        choices=[0, 1, 2],
        default=0,
        help='Model type: 0=exponential, 1=powerlaw, 2=rayleigh'
    )

    parser.add_argument(
        '--max-iter', 
        type=int,
        default=2000,
        help='Maximum number of iterations'
    )

    parser.add_argument(
        '--name-output', 
        type=str,
        default='inferred-network',
        help='Base name for output files'
    )

    parser.add_argument(
        '--all-models', 
        action='store_true',
        help='Run inference for all three models (ignores --model parameter)'
    )

    args = parser.parse_args()

    # Print configuration
    print("Network Inference Configuration:")
    print(f"  Cascades file: {args.cascades}")
    print(f"  Alpha range: {args.alpha_0:.2e} to {args.alpha_f:.2e}")
    print(f"  Number of alphas: {args.N}")
    print(f"  Max iterations: {args.max_iter}")
    print(f"  Output name: {args.name_output}")

    if args.all_models:
        print("  Models: All (exponential, powerlaw, rayleigh)")
        print("\nStarting inference for all models...")

        results = infer_networks_all_models(
            cascades_file=args.cascades,
            alpha_0=args.alpha_0,
            alpha_f=args.alpha_f,
            n=args.N,
            max_iter=args.max_iter,
            name_output=args.name_output
        )

        # Print final summary
        all_success = all(results.values())
        if all_success:
            print("\n✓ All models completed successfully!")
            sys.exit(0)
        else:
            print("\n✗ Some models failed. Check the output above for details.")
            sys.exit(1)

    else:
        model_names = {0: "exponential", 1: "powerlaw", 2: "rayleigh"}
        print(f"  Model: {model_names[args.model]} ({args.model})")
        print(f"\nStarting inference for {model_names[args.model]} model...")

        success = infer_networks(
            cascades_file=args.cascades,
            alpha_0=args.alpha_0,
            alpha_f=args.alpha_f,
            n=args.N,
            model=args.model,
            max_iter=args.max_iter,
            name_output=args.name_output
        )

        if success:
            print("\n✓ Inference completed successfully!")
            sys.exit(0)
        else:
            print("\n✗ Inference failed. Check the output above for details.")
            sys.exit(1)

if __name__ == '__main__':
    main()
