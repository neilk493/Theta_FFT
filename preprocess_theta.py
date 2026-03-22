"""
preprocess_theta.py

Takes theta adjacent_angles.csv for a protein,
outputs a clean FFT-ready CSV with sequential residue index and theta_signed.

Usage:
    python preprocess_theta.py \
        --angles 1TEN_boxes_adjacent_angles.csv \
        --output 1TEN_fft_data.csv \
        --protein 1TEN
"""

import argparse
import pandas as pd


def load_angles(path):
    df = pd.read_csv(path)
    out = df[['chain', 'res_i_A', 'res_j_A', 'angle_signed_deg']].copy()
    out.columns = ['chain', 'res_i', 'res_j', 'theta_signed']
    return out


def build_series(angles_df, chain=None):
    if chain:
        angles_df = angles_df[angles_df['chain'] == chain].copy()
    else:
        chain = angles_df['chain'].iloc[0]
        angles_df = angles_df[angles_df['chain'] == chain].copy()

    merged = angles_df.sort_values('res_i').reset_index(drop=True)

    res_diffs = merged['res_i'].diff()
    gaps = res_diffs[res_diffs > 1]
    if len(gaps) > 0:
        print(f"  Warning: {len(gaps)} gap(s) detected in residue numbering.")

    merged['seq_index'] = range(len(merged))
    merged['has_gap_before'] = (res_diffs > 1).fillna(0).astype(int)

    return merged[['seq_index', 'res_i', 'res_j', 'theta_signed', 'has_gap_before']], chain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--angles', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--protein', default='protein')
    parser.add_argument('--chain', default=None)
    args = parser.parse_args()

    print(f"\nProcessing {args.protein}...")

    angles_df = load_angles(args.angles)
    series, chain_used = build_series(angles_df, chain=args.chain)
    series['protein'] = args.protein
    series['chain'] = chain_used

    theta = series['theta_signed'].dropna().values
    print(f"  Chain: {chain_used}, Residues: {len(series)}, Theta range: {theta.min():.1f} to {theta.max():.1f}")

    series = series[['protein', 'chain', 'seq_index', 'res_i', 'res_j', 'theta_signed', 'has_gap_before']]
    series.to_csv(args.output, index=False)
    print(f"  Saved to {args.output}")


if __name__ == '__main__':
    main()