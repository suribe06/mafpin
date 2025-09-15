#!/usr/bin/env python3
"""
Centrality Metrics Runner Script

This script provides a command-line interface to calculate centrality metrics
for inferred networks using SNAP-py.

Usage examples:
- python run_centrality_metrics.py --help
- python run_centrality_metrics.py --single-network network_file.txt --plots
- python run_centrality_metrics.py --all-models --plots
- python run_centrality_metrics.py --model exponential --plots
"""

import argparse
import glob
import os
import sys

from calculate_centrality_metrics import (
    calculate_centrality_for_all_models,
    calculate_centrality_for_network,
)

def main():
    """Main function to parse arguments and run centrality metrics calculation."""

    parser = argparse.ArgumentParser(
        description='Calculate centrality metrics for inferred networks using SNAP-py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Network selection options
    network_group = parser.add_mutually_exclusive_group(required=True)

    network_group.add_argument(
        '--single-network',
        type=str,
        help='Process a single network file'
    )

    network_group.add_argument(
        '--all-models',
        action='store_true',
        help='Process all networks from all models (exponential, powerlaw, rayleigh)'
    )

    network_group.add_argument(
        '--model',
        type=str,
        choices=['exponential', 'powerlaw', 'rayleigh'],
        help='Process all networks from a specific model'
    )

    # Options
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate distribution plots for centrality metrics'
    )

    parser.add_argument(
        '--base-path',
        type=str,
        default='../data/inferred_networks',
        help='Base path to inferred networks directory'
    )

    args = parser.parse_args()

    print("Centrality Metrics Calculator")
    print("="*50)

    if args.single_network:
        print(f"Processing single network: {args.single_network}")
        print(f"Plots enabled: {args.plots}")

        if not os.path.exists(args.single_network):
            print(f"Error: Network file '{args.single_network}' not found.")
            sys.exit(1)

        success = calculate_centrality_for_network(args.single_network, args.plots)

        if success:
            print("\n✓ Network processing completed successfully!")
            sys.exit(0)
        else:
            print("\n✗ Network processing failed.")
            sys.exit(1)

    elif args.all_models:
        print("Processing all models (exponential, powerlaw, rayleigh)")
        print(f"Plots enabled: {args.plots}")
        print(f"Base path: {args.base_path}")

        if not os.path.exists(args.base_path):
            print(f"Error: Base path '{args.base_path}' not found.")
            sys.exit(1)

        results = calculate_centrality_for_all_models(args.plots)

        # Check results
        total_processed = sum(r.get('processed', 0) for r in results.values())
        total_failed = sum(r.get('failed', 0) for r in results.values())

        if total_processed > 0:
            print(f"\n✓ Processing completed! {total_processed} networks processed, {total_failed} failed.")
            sys.exit(0)
        else:
            print(f"\n✗ No networks were processed successfully. {total_failed} failed.")
            sys.exit(1)

    elif args.model:
        print(f"Processing model: {args.model}")
        print(f"Plots enabled: {args.plots}")

        model_path = os.path.join(args.base_path, args.model)

        if not os.path.exists(model_path):
            print(f"Error: Model directory '{model_path}' not found.")
            sys.exit(1)

        # Find all network files for this model
        pattern = os.path.join(model_path, '*.txt')
        network_files = glob.glob(pattern)

        # Filter out CSV files
        network_files = [f for f in network_files if not f.endswith('.csv')]

        if not network_files:
            print(f"Error: No network files found in '{model_path}'")
            sys.exit(1)

        print(f"Found {len(network_files)} network files")

        processed = 0
        failed = 0

        for network_file in sorted(network_files):
            try:
                success = calculate_centrality_for_network(network_file, args.plots)
                if success:
                    processed += 1
                else:
                    failed += 1
            except Exception as e: # pylint: disable=broad-except
                print(f"Error processing {network_file}: {str(e)}")
                failed += 1

        print(f"\nModel {args.model} summary: {processed} processed, {failed} failed")

        if processed > 0:
            print(f"\n✓ Model processing completed! {processed} networks processed.")
            sys.exit(0)
        else:
            print(f"\n✗ No networks were processed successfully. {failed} failed.")
            sys.exit(1)

if __name__ == '__main__':
    main()
