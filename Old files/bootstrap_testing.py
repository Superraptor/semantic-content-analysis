#!/usr/bin/env python

#
#   Clair Kronk
#   30 October 2024
#   bootstrap_testing.py
#
#   Bootstrap testing script - runs OFFLINE paralinguistic analysis 100 times
#   and aggregates results into a single Excel file
#

import argparse
import logging
import pandas as pd
import sys
import time
from pathlib import Path

# Import functions from OFFLINE script
try:
    from OFFLINE_perform_paralinguistic_analysis import (
        load_models,
        process_single_file,
        extract_results_as_dict
    )
except ImportError as e:
    print(f"Error: Could not import from OFFLINE_perform_paralinguistic_analysis.py: {e}")
    print("Make sure OFFLINE_perform_paralinguistic_analysis.py is in the same directory.")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_bootstrap_analysis(audio_file, num_iterations=100, skip_diarization=False, max_duration=None):
    """
    Run paralinguistic analysis multiple times and collect results

    Args:
        audio_file (str): Path to audio file
        num_iterations (int): Number of iterations to run (default=100)
        skip_diarization (bool): Whether to skip speaker diarization
        max_duration (float): Maximum audio duration in seconds

    Returns:
        pandas.DataFrame: DataFrame with all results
    """
    logging.info(f"Starting bootstrap analysis with {num_iterations} iterations")
    logging.info(f"Audio file: {audio_file}")

    # Ask user once about speaker segment reuse
    print()
    print("=" * 70)
    print("SPEAKER SEGMENT FILE HANDLING")
    print("=" * 70)
    print("The script will check if speaker segment files already exist from a")
    print("previous run (e.g., converted_audio_speaker_SPEAKER_00.wav).")
    print()
    print("Options:")
    print("  - REUSE (recommended): Use existing files, saves significant time")
    print("  - REGENERATE: Create new speaker segments for each iteration")
    print()
    user_input = input("Would you like to REUSE existing speaker segment files? (Y/n): ").strip().lower()

    if user_input in ['n', 'no', 'false', 'f']:
        auto_reuse_segments = False
        print("➜ Will REGENERATE speaker segment files for each iteration")
    else:
        auto_reuse_segments = True
        print("➜ Will REUSE existing speaker segment files (faster)")

    print("=" * 70)
    print()

    # Load models once (shared across all iterations)
    logging.info("Loading models (one-time setup)...")
    load_models()
    logging.info("Models loaded successfully")

    # Collect all rows
    all_rows = []
    start_time = time.time()

    for iteration in range(1, num_iterations + 1):
        logging.info(f"Running iteration {iteration}/{num_iterations}")
        iteration_start = time.time()

        try:
            # Run analysis for this iteration
            result = process_single_file(
                audio_file,
                skip_diarization=skip_diarization,
                max_duration=max_duration,
                auto_reuse_segments=auto_reuse_segments
            )

            # Extract results
            speaker_analysis = result['speaker_analysis']
            comprehension = result.get('comprehension', {})

            # Convert to dictionary rows
            rows = extract_results_as_dict(
                speaker_analysis,
                comprehension_summary=comprehension,
                iteration_id=iteration
            )

            all_rows.extend(rows)

            iteration_time = time.time() - iteration_start
            logging.info(f"Iteration {iteration} completed in {iteration_time:.2f} seconds "
                        f"({len(rows)} speakers detected)")

        except Exception as e:
            logging.error(f"Error in iteration {iteration}: {e}")
            logging.info(f"Skipping iteration {iteration} and continuing...")
            continue

    total_time = time.time() - start_time
    logging.info(f"Bootstrap analysis completed in {total_time:.2f} seconds")
    logging.info(f"Total iterations: {num_iterations}")
    logging.info(f"Total rows collected: {len(all_rows)}")
    logging.info(f"Average time per iteration: {total_time/num_iterations:.2f} seconds")

    # Convert to DataFrame
    df = pd.DataFrame(all_rows)

    return df

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap testing - Run OFFLINE paralinguistic analysis multiple times"
    )
    parser.add_argument('audiofile', type=str, help='Path to audio file')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations to run (default: 100)')
    parser.add_argument('--output', type=str, default='bootstrap_output.xlsx',
                       help='Output Excel file name (default: bootstrap_output.xlsx)')
    parser.add_argument('--skip-diarization', action='store_true',
                       help='Skip speaker diarization')
    parser.add_argument('--max-duration', type=float, default=None,
                       help='Max audio duration (seconds)')

    args = parser.parse_args()

    # Verify audio file exists
    if not Path(args.audiofile).exists():
        print(f"Error: Audio file '{args.audiofile}' not found")
        sys.exit(1)

    print("=" * 70)
    print("BOOTSTRAP TESTING - OFFLINE PARALINGUISTIC ANALYSIS")
    print("=" * 70)
    print(f"Audio file: {args.audiofile}")
    print(f"Iterations: {args.iterations}")
    print(f"Output file: {args.output}")
    print("=" * 70)
    print()

    # Run bootstrap analysis
    df = run_bootstrap_analysis(
        args.audiofile,
        num_iterations=args.iterations,
        skip_diarization=args.skip_diarization,
        max_duration=args.max_duration
    )

    # Save to Excel
    output_file = Path(args.output)
    logging.info(f"Saving results to {output_file}")

    try:
        df.to_excel(output_file, index=False, engine='openpyxl')
        print()
        print("=" * 70)
        print("BOOTSTRAP TESTING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Output file: {output_file}")
        print(f"Total rows: {len(df)}")
        print(f"Iterations: {args.iterations}")
        print(f"Speakers per iteration (avg): {len(df)/args.iterations:.2f}")
        print()

        # Display summary statistics
        print("SUMMARY STATISTICS:")
        print("-" * 70)

        # Numeric columns to summarize
        numeric_cols = ['Pitch_Mean', 'Pitch_Std', 'Intensity_Mean', 'Intensity_Std',
                       'MFCC_Mean', 'MFCC_Std', 'Tempo', 'Overall_Similarity_Score',
                       'Overall_Comprehension_Score', 'Speaker1_Sentiment',
                       'Speaker2_Sentiment', 'Sentiment_Difference']

        for col in numeric_cols:
            if col in df.columns:
                print(f"{col:30s}: Mean={df[col].mean():8.4f}, Std={df[col].std():8.4f}")

        print()
        print("Emotion distribution:")
        print(df['Emotion'].value_counts())
        print()

    except Exception as e:
        logging.error(f"Error saving Excel file: {e}")
        print(f"\nError: Could not save Excel file: {e}")
        print("Make sure you have openpyxl installed: pip install openpyxl")
        sys.exit(1)

if __name__ == "__main__":
    main()
