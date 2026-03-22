"""Sliding-window local spectral analysis for contiguous theta signal segments.

This script performs sliding-window local spectral analysis on contiguous theta
signal segments and computes FFT-based numeric outputs only. By design it is
separate from global analysis, plotting, comparison, and validation, and it
produces spectrogram-ready long tables plus window feature tables without
performing new preprocessing, classification, or plotting.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: list[str] = [
    "protein",
    "chain",
    "seq_index",
    "res_i",
    "res_j",
    "theta_signed",
    "has_gap_before",
]

WINDOW_METADATA_COLUMNS: list[str] = [
    "protein",
    "chain",
    "source_file",
    "segment_id",
    "segment_label",
    "segment_start_seq_index",
    "segment_end_seq_index",
    "segment_length",
    "segment_start_res_i",
    "segment_end_res_j",
    "window_id",
    "window_label",
    "window_index_within_segment",
    "window_start_offset_in_segment",
    "window_end_offset_in_segment",
    "window_start_seq_index",
    "window_end_seq_index",
    "window_center_seq_index",
    "window_start_res_i",
    "window_end_res_j",
    "window_length",
]

SPECTRUM_COLUMNS: list[str] = WINDOW_METADATA_COLUMNS + [
    "fft_bin",
    "frequency_cycles_per_residue",
    "period_residues",
    "amplitude",
    "power",
    "power_fraction_nonzero",
]

FEATURE_COLUMNS: list[str] = WINDOW_METADATA_COLUMNS + [
    "theta_mean_deg",
    "theta_std_deg",
    "theta_min_deg",
    "theta_max_deg",
    "n_fft_bins",
    "total_nonzero_power",
    "dominant_frequency_cycles_per_residue",
    "dominant_period_residues",
    "dominant_power",
    "dominant_power_fraction_nonzero",
    "spectral_centroid_cycles_per_residue",
    "spectral_bandwidth_cycles_per_residue",
    "peak1_frequency_cycles_per_residue",
    "peak1_period_residues",
    "peak1_power",
    "peak1_power_fraction_nonzero",
    "peak2_frequency_cycles_per_residue",
    "peak2_period_residues",
    "peak2_power",
    "peak2_power_fraction_nonzero",
    "peak3_frequency_cycles_per_residue",
    "peak3_period_residues",
    "peak3_power",
    "peak3_power_fraction_nonzero",
    "low_band_power",
    "mid_band_power",
    "high_band_power",
    "low_band_power_fraction",
    "mid_band_power_fraction",
    "high_band_power_fraction",
    "relative_window_start_in_segment",
    "relative_window_end_in_segment",
    "relative_window_center_in_segment",
]

SKIPPED_COLUMNS: list[str] = [
    "protein",
    "chain",
    "source_file",
    "segment_id",
    "segment_label",
    "segment_start_seq_index",
    "segment_end_seq_index",
    "segment_length",
    "segment_start_res_i",
    "segment_end_res_j",
    "reason",
]

BAND_DEFINITIONS: dict[str, dict[str, float | str]] = {
    "low": {"range": "0 < f <= 1/12", "lower_exclusive": 0.0, "upper_inclusive": 1.0 / 12.0},
    "mid": {
        "range": "1/12 < f <= 1/6",
        "lower_exclusive": 1.0 / 12.0,
        "upper_inclusive": 1.0 / 6.0,
    },
    "high": {
        "range": "1/6 < f <= 1/2",
        "lower_exclusive": 1.0 / 6.0,
        "upper_inclusive": 1.0 / 2.0,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for local spectral analysis."""

    parser = argparse.ArgumentParser(
        description="Compute sliding-window local FFT outputs for contiguous theta segments."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Theta_FFT/fft_data"),
        help="Directory containing *_fft_data.csv inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Theta_FFT/output/spectrograms"),
        help="Directory for spectrogram-ready numeric outputs.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="Sliding window length in residues.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=1,
        help="Sliding window stride in residues.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file, per-segment, and per-window summaries.",
    )
    return parser.parse_args()


def ensure_output_dir(output_dir: Path) -> None:
    """Create the output directory if it does not already exist."""

    output_dir.mkdir(parents=True, exist_ok=True)


def discover_input_files(input_dir: Path) -> list[Path]:
    """Discover all input CSVs matching the expected local FFT pattern."""

    csv_files = sorted(input_dir.glob("*_fft_data.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_fft_data.csv files found in input directory: {input_dir}")
    return csv_files


def validate_input_df(df: pd.DataFrame, source_name: str) -> None:
    """Validate required columns and core data types for one input CSV."""

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{source_name}: missing required columns: {missing}")

    if df.empty:
        raise ValueError(f"{source_name}: input CSV is empty")

    if df["protein"].isna().any() or (df["protein"].astype(str).str.strip() == "").any():
        raise ValueError(f"{source_name}: protein must not be missing")
    if df["chain"].isna().any() or (df["chain"].astype(str).str.strip() == "").any():
        raise ValueError(f"{source_name}: chain must not be missing")

    theta_numeric = pd.to_numeric(df["theta_signed"], errors="coerce")
    if theta_numeric.isna().any():
        raise ValueError(f"{source_name}: theta_signed must be numeric")

    seq_numeric = pd.to_numeric(df["seq_index"], errors="coerce")
    if seq_numeric.isna().any():
        raise ValueError(f"{source_name}: seq_index must be integer-like")
    seq_values = seq_numeric.to_numpy(dtype=float)
    if not np.all(np.isclose(seq_values, np.round(seq_values))):
        raise ValueError(f"{source_name}: seq_index must be integer-like")

    gap_numeric = pd.to_numeric(df["has_gap_before"], errors="coerce")
    if gap_numeric.isna().any():
        raise ValueError(f"{source_name}: has_gap_before must contain only 0 or 1")
    gap_values = gap_numeric.to_numpy(dtype=float)
    if not np.all(np.isclose(gap_values, np.round(gap_values))):
        raise ValueError(f"{source_name}: has_gap_before must contain only 0 or 1")
    if set(gap_numeric.astype(int).tolist()) - {0, 1}:
        raise ValueError(f"{source_name}: has_gap_before must contain only 0 or 1")


def split_chain_into_segments(chain_df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a validated protein-chain table into contiguous observed segments."""

    if chain_df.empty:
        return []

    segment_start_flags = np.zeros(len(chain_df), dtype=bool)
    segment_start_flags[0] = True
    if len(chain_df) > 1:
        segment_start_flags[1:] = chain_df["has_gap_before"].astype(int).to_numpy()[1:] == 1

    segment_numbers = np.cumsum(segment_start_flags) - 1
    segments: list[pd.DataFrame] = []
    for segment_number in np.unique(segment_numbers):
        segment_df = chain_df.loc[segment_numbers == segment_number].copy().reset_index(drop=True)
        segments.append(segment_df)
    return segments


def generate_windows(segment_df: pd.DataFrame, window_size: int, step_size: int) -> list[pd.DataFrame]:
    """Generate all full sliding windows for one contiguous observed segment."""

    segment_length = len(segment_df)
    if segment_length < window_size:
        return []

    windows: list[pd.DataFrame] = []
    last_start = segment_length - window_size
    for start in range(0, last_start + 1, step_size):
        window_df = segment_df.iloc[start : start + window_size].copy().reset_index(drop=True)
        windows.append(window_df)
    return windows


def compute_local_spectrum(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the real FFT spectrum for one mean-centered theta window."""

    centered = theta.astype(float) - float(np.mean(theta))
    n = centered.size
    freqs = np.fft.rfftfreq(n, d=1.0)
    fft_vals = np.fft.rfft(centered)
    amplitude = np.abs(fft_vals)
    power = (np.abs(fft_vals) ** 2) / n
    return freqs, fft_vals, amplitude, power


def build_window_spectrum_table(
    window_df: pd.DataFrame,
    source_file: str,
    segment_id: str,
    window_id: str,
) -> pd.DataFrame:
    """Build the long-format local spectrum table for one sliding window."""

    first_row = window_df.iloc[0]
    last_row = window_df.iloc[-1]

    protein = str(first_row["protein"])
    chain = str(first_row["chain"])
    segment_label = f"{protein}_{chain}_{segment_id}"
    window_label = f"{segment_label}_{window_id}"

    segment_start_seq_index = int(window_df.attrs["segment_start_seq_index"])
    segment_end_seq_index = int(window_df.attrs["segment_end_seq_index"])
    segment_length = int(window_df.attrs["segment_length"])
    segment_start_res_i = int(window_df.attrs["segment_start_res_i"])
    segment_end_res_j = int(window_df.attrs["segment_end_res_j"])
    window_index_within_segment = int(window_df.attrs["window_index_within_segment"])
    window_start_offset = int(window_df.attrs["window_start_offset_in_segment"])
    window_end_offset = int(window_df.attrs["window_end_offset_in_segment"])

    theta = window_df["theta_signed"].to_numpy(dtype=float)
    freqs, fft_vals, amplitude, power = compute_local_spectrum(theta)

    nonzero_mask = freqs > 0
    total_nonzero_power = float(power[nonzero_mask].sum())
    power_fraction_nonzero = np.full(freqs.shape, np.nan, dtype=float)
    if total_nonzero_power > 0.0:
        power_fraction_nonzero[nonzero_mask] = power[nonzero_mask] / total_nonzero_power

    period_residues = np.full(freqs.shape, np.nan, dtype=float)
    period_residues[nonzero_mask] = 1.0 / freqs[nonzero_mask]

    window_length = int(len(window_df))
    spectrum_df = pd.DataFrame(
        {
            "protein": protein,
            "chain": chain,
            "source_file": source_file,
            "segment_id": segment_id,
            "segment_label": segment_label,
            "segment_start_seq_index": segment_start_seq_index,
            "segment_end_seq_index": segment_end_seq_index,
            "segment_length": segment_length,
            "segment_start_res_i": segment_start_res_i,
            "segment_end_res_j": segment_end_res_j,
            "window_id": window_id,
            "window_label": window_label,
            "window_index_within_segment": window_index_within_segment,
            "window_start_offset_in_segment": window_start_offset,
            "window_end_offset_in_segment": window_end_offset,
            "window_start_seq_index": int(first_row["seq_index"]),
            "window_end_seq_index": int(last_row["seq_index"]),
            "window_center_seq_index": float(
                int(first_row["seq_index"]) + (window_length - 1) / 2.0
            ),
            "window_start_res_i": int(first_row["res_i"]),
            "window_end_res_j": int(last_row["res_j"]),
            "window_length": window_length,
            "fft_bin": np.arange(freqs.size, dtype=int),
            "frequency_cycles_per_residue": freqs.astype(float),
            "period_residues": period_residues,
            "amplitude": amplitude.astype(float),
            "power": power.astype(float),
            "power_fraction_nonzero": power_fraction_nonzero,
        }
    )
    _ = fft_vals
    return spectrum_df.loc[:, SPECTRUM_COLUMNS]


def extract_window_features(spectrum_df: pd.DataFrame, original_theta: np.ndarray) -> dict:
    """Extract per-window spectral summary features from one window spectrum table."""

    metadata = {column: spectrum_df.iloc[0][column] for column in WINDOW_METADATA_COLUMNS}

    theta_array = original_theta.astype(float)
    nonzero_df = spectrum_df.loc[spectrum_df["frequency_cycles_per_residue"] > 0].copy()
    total_nonzero_power = float(nonzero_df["power"].sum())

    dominant_frequency = np.nan
    dominant_period = np.nan
    dominant_power = np.nan
    dominant_power_fraction = np.nan
    spectral_centroid = np.nan
    spectral_bandwidth = np.nan

    if not nonzero_df.empty and total_nonzero_power > 0.0:
        dominant_row = nonzero_df.sort_values(
            by=["power", "frequency_cycles_per_residue"],
            ascending=[False, True],
        ).iloc[0]
        dominant_frequency = float(dominant_row["frequency_cycles_per_residue"])
        dominant_period = float(dominant_row["period_residues"])
        dominant_power = float(dominant_row["power"])
        dominant_power_fraction = float(dominant_row["power_fraction_nonzero"])

        freq_values = nonzero_df["frequency_cycles_per_residue"].to_numpy(dtype=float)
        power_values = nonzero_df["power"].to_numpy(dtype=float)
        spectral_centroid = float(np.sum(freq_values * power_values) / total_nonzero_power)
        spectral_bandwidth = float(
            np.sqrt(np.sum(power_values * (freq_values - spectral_centroid) ** 2) / total_nonzero_power)
        )

    peak_rows = nonzero_df.sort_values(
        by=["power", "frequency_cycles_per_residue"],
        ascending=[False, True],
    ).head(3)

    peak_features: dict[str, float] = {}
    for rank in range(1, 4):
        if rank <= len(peak_rows):
            peak_row = peak_rows.iloc[rank - 1]
            peak_features[f"peak{rank}_frequency_cycles_per_residue"] = float(
                peak_row["frequency_cycles_per_residue"]
            )
            peak_features[f"peak{rank}_period_residues"] = float(peak_row["period_residues"])
            peak_features[f"peak{rank}_power"] = float(peak_row["power"])
            peak_features[f"peak{rank}_power_fraction_nonzero"] = float(
                peak_row["power_fraction_nonzero"]
            )
        else:
            peak_features[f"peak{rank}_frequency_cycles_per_residue"] = np.nan
            peak_features[f"peak{rank}_period_residues"] = np.nan
            peak_features[f"peak{rank}_power"] = np.nan
            peak_features[f"peak{rank}_power_fraction_nonzero"] = np.nan

    freq_values = nonzero_df["frequency_cycles_per_residue"]
    low_mask = (freq_values > 0.0) & (freq_values <= 1.0 / 12.0)
    mid_mask = (freq_values > 1.0 / 12.0) & (freq_values <= 1.0 / 6.0)
    high_mask = (freq_values > 1.0 / 6.0) & (freq_values <= 1.0 / 2.0)

    low_band_power = float(nonzero_df.loc[low_mask, "power"].sum())
    mid_band_power = float(nonzero_df.loc[mid_mask, "power"].sum())
    high_band_power = float(nonzero_df.loc[high_mask, "power"].sum())

    def safe_fraction(value: float) -> float:
        if total_nonzero_power > 0.0:
            return float(value / total_nonzero_power)
        return np.nan

    segment_length = float(metadata["segment_length"])
    window_start_offset = float(metadata["window_start_offset_in_segment"])
    window_end_offset = float(metadata["window_end_offset_in_segment"])
    window_length = float(metadata["window_length"])

    features: dict[str, Any] = {
        **metadata,
        "theta_mean_deg": float(np.mean(theta_array)),
        "theta_std_deg": float(np.std(theta_array, ddof=0)),
        "theta_min_deg": float(np.min(theta_array)),
        "theta_max_deg": float(np.max(theta_array)),
        "n_fft_bins": int(len(spectrum_df)),
        "total_nonzero_power": total_nonzero_power,
        "dominant_frequency_cycles_per_residue": dominant_frequency,
        "dominant_period_residues": dominant_period,
        "dominant_power": dominant_power,
        "dominant_power_fraction_nonzero": dominant_power_fraction,
        "spectral_centroid_cycles_per_residue": spectral_centroid,
        "spectral_bandwidth_cycles_per_residue": spectral_bandwidth,
        **peak_features,
        "low_band_power": low_band_power,
        "mid_band_power": mid_band_power,
        "high_band_power": high_band_power,
        "low_band_power_fraction": safe_fraction(low_band_power),
        "mid_band_power_fraction": safe_fraction(mid_band_power),
        "high_band_power_fraction": safe_fraction(high_band_power),
        "relative_window_start_in_segment": float(window_start_offset / segment_length),
        "relative_window_end_in_segment": float((window_end_offset + 1.0) / segment_length),
        "relative_window_center_in_segment": float(
            (window_start_offset + (window_length - 1.0) / 2.0) / segment_length
        ),
    }
    return features


def process_file(
    csv_path: Path,
    window_size: int,
    step_size: int,
    verbose: bool = False,
) -> tuple[list[pd.DataFrame], list[dict], list[dict]]:
    """Process one input CSV into per-segment local spectra, window features, and skipped rows."""

    df = pd.read_csv(csv_path)
    validate_input_df(df, csv_path.name)

    df = df.copy()
    df["protein"] = df["protein"].astype(str)
    df["chain"] = df["chain"].astype(str)
    df["seq_index"] = pd.to_numeric(df["seq_index"], errors="raise").astype(int)
    df["res_i"] = pd.to_numeric(df["res_i"], errors="raise").astype(int)
    df["res_j"] = pd.to_numeric(df["res_j"], errors="raise").astype(int)
    df["theta_signed"] = pd.to_numeric(df["theta_signed"], errors="raise").astype(float)
    df["has_gap_before"] = pd.to_numeric(df["has_gap_before"], errors="raise").astype(int)

    spectra_tables: list[pd.DataFrame] = []
    feature_rows: list[dict] = []
    skipped_rows: list[dict] = []

    grouped = df.groupby(["protein", "chain"], sort=True, dropna=False)
    for (protein, chain), chain_df in grouped:
        sorted_chain = chain_df.sort_values("seq_index", kind="stable").reset_index(drop=True)

        if (sorted_chain["seq_index"] < 0).any():
            raise ValueError(f"{csv_path.name}: seq_index must be nonnegative for {protein}/{chain}")
        if sorted_chain["seq_index"].duplicated().any():
            raise ValueError(f"{csv_path.name}: duplicate seq_index detected for {protein}/{chain}")

        seq_diffs = np.diff(sorted_chain["seq_index"].to_numpy(dtype=int))
        if seq_diffs.size > 0 and not np.all(seq_diffs == 1):
            raise ValueError(
                f"{csv_path.name}: seq_index must increase by exactly 1 after sorting for {protein}/{chain}"
            )

        segments = split_chain_into_segments(sorted_chain)
        if verbose:
            print(f"Processing {csv_path.name}: protein={protein}, chain={chain}, segments={len(segments)}")

        for segment_number, segment_df in enumerate(segments):
            segment_id = f"seg{segment_number:03d}"
            segment_label = f"{protein}_{chain}_{segment_id}"
            first_row = segment_df.iloc[0]
            last_row = segment_df.iloc[-1]
            segment_length = int(len(segment_df))

            segment_metadata: dict[str, Any] = {
                "protein": protein,
                "chain": chain,
                "source_file": csv_path.name,
                "segment_id": segment_id,
                "segment_label": segment_label,
                "segment_start_seq_index": int(first_row["seq_index"]),
                "segment_end_seq_index": int(last_row["seq_index"]),
                "segment_length": segment_length,
                "segment_start_res_i": int(first_row["res_i"]),
                "segment_end_res_j": int(last_row["res_j"]),
            }

            if segment_length < window_size:
                skipped_rows.append(
                    {
                        **segment_metadata,
                        "reason": "segment_shorter_than_window_size",
                    }
                )
                if verbose:
                    print(
                        f"  Skipping {segment_label}: length={segment_length} < window_size={window_size}"
                    )
                continue

            windows = generate_windows(segment_df, window_size=window_size, step_size=step_size)
            if verbose:
                print(
                    f"  Analyzing {segment_label}: length={segment_length}, windows={len(windows)}"
                )

            segment_window_tables: list[pd.DataFrame] = []
            for window_index, window_df in enumerate(windows):
                window_id = f"win{window_index:05d}"
                window_start_offset = window_index * step_size
                window_end_offset = window_start_offset + window_size - 1

                window_df.attrs["segment_start_seq_index"] = segment_metadata["segment_start_seq_index"]
                window_df.attrs["segment_end_seq_index"] = segment_metadata["segment_end_seq_index"]
                window_df.attrs["segment_length"] = segment_metadata["segment_length"]
                window_df.attrs["segment_start_res_i"] = segment_metadata["segment_start_res_i"]
                window_df.attrs["segment_end_res_j"] = segment_metadata["segment_end_res_j"]
                window_df.attrs["window_index_within_segment"] = window_index
                window_df.attrs["window_start_offset_in_segment"] = window_start_offset
                window_df.attrs["window_end_offset_in_segment"] = window_end_offset

                spectrum_df = build_window_spectrum_table(
                    window_df=window_df,
                    source_file=csv_path.name,
                    segment_id=segment_id,
                    window_id=window_id,
                )
                segment_window_tables.append(spectrum_df)
                feature_rows.append(
                    extract_window_features(
                        spectrum_df=spectrum_df,
                        original_theta=window_df["theta_signed"].to_numpy(dtype=float),
                    )
                )

                if verbose:
                    window_start = int(window_df.iloc[0]["seq_index"])
                    window_end = int(window_df.iloc[-1]["seq_index"])
                    print(
                        f"    Window {window_id}: start_seq_index={window_start}, end_seq_index={window_end}"
                    )

            segment_spectrum_df = pd.concat(segment_window_tables, ignore_index=True)
            segment_spectrum_df = segment_spectrum_df.sort_values(
                by=[
                    "window_start_seq_index",
                    "window_index_within_segment",
                    "frequency_cycles_per_residue",
                    "fft_bin",
                ],
                kind="stable",
            ).reset_index(drop=True)
            spectra_tables.append(segment_spectrum_df.loc[:, SPECTRUM_COLUMNS])

    return spectra_tables, feature_rows, skipped_rows


def write_outputs(
    output_dir: Path,
    spectrum_tables: list[pd.DataFrame],
    feature_rows: list[dict],
    skipped_rows: list[dict],
    metadata: dict[str, Any],
) -> dict[str, Path]:
    """Write combined and per-segment outputs plus skipped rows and run metadata."""

    ensure_output_dir(output_dir)

    long_output_path = output_dir / "local_spectra_long.csv"
    features_output_path = output_dir / "local_window_features.csv"
    skipped_output_path = output_dir / "skipped_segments.csv"
    metadata_output_path = output_dir / "run_metadata.json"

    if spectrum_tables:
        combined_spectrum_df = pd.concat(spectrum_tables, ignore_index=True)
        combined_spectrum_df = combined_spectrum_df.sort_values(
            by=[
                "protein",
                "chain",
                "segment_start_seq_index",
                "window_start_seq_index",
                "frequency_cycles_per_residue",
                "fft_bin",
            ],
            kind="stable",
        ).reset_index(drop=True)
        combined_spectrum_df.loc[:, SPECTRUM_COLUMNS].to_csv(long_output_path, index=False)

        for segment_df in spectrum_tables:
            segment_label = str(segment_df.iloc[0]["segment_label"])
            segment_path = output_dir / f"{segment_label}_local_spectra_long.csv"
            segment_df.loc[:, SPECTRUM_COLUMNS].to_csv(segment_path, index=False)
    else:
        pd.DataFrame(columns=SPECTRUM_COLUMNS).to_csv(long_output_path, index=False)

    features_df = pd.DataFrame(feature_rows)
    if features_df.empty:
        features_df = pd.DataFrame(columns=FEATURE_COLUMNS)
    else:
        features_df = features_df.loc[:, FEATURE_COLUMNS].sort_values(
            by=[
                "protein",
                "chain",
                "segment_start_seq_index",
                "window_start_seq_index",
                "window_index_within_segment",
            ],
            kind="stable",
        ).reset_index(drop=True)
    features_df.to_csv(features_output_path, index=False)

    skipped_df = pd.DataFrame(skipped_rows)
    if skipped_df.empty:
        skipped_df = pd.DataFrame(columns=SKIPPED_COLUMNS)
    else:
        skipped_df = skipped_df.loc[:, SKIPPED_COLUMNS].sort_values(
            by=["protein", "chain", "segment_start_seq_index"],
            kind="stable",
        ).reset_index(drop=True)
    skipped_df.to_csv(skipped_output_path, index=False)

    with metadata_output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "local_spectra_long": long_output_path,
        "local_window_features": features_output_path,
        "skipped_segments": skipped_output_path,
        "run_metadata": metadata_output_path,
    }


def resolve_cli_path(path_value: Path, script_dir: Path) -> Path:
    """Resolve a CLI path robustly from the project root or its parent directory."""

    if path_value.is_absolute():
        return path_value

    candidates = [
        Path.cwd() / path_value,
        script_dir / path_value,
        script_dir.parent / path_value,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if path_value.parts and path_value.parts[0] == script_dir.name:
        return (script_dir.parent / path_value).resolve()
    return (script_dir / path_value).resolve()


def main() -> None:
    """Run sliding-window local FFT analysis across all discovered input files."""

    args = parse_args()
    if args.window_size < 2:
        raise ValueError("--window-size must be >= 2")
    if args.step_size < 1:
        raise ValueError("--step-size must be >= 1")

    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    input_dir = resolve_cli_path(args.input_dir, script_dir)
    output_dir = resolve_cli_path(args.output_dir, script_dir)

    input_files = discover_input_files(input_dir)
    print(f"Found {len(input_files)} input files in {input_dir}")

    all_spectrum_tables: list[pd.DataFrame] = []
    all_feature_rows: list[dict] = []
    all_skipped_rows: list[dict] = []

    for csv_path in input_files:
        if args.verbose:
            print(f"Reading {csv_path}")
        spectra_tables, feature_rows, skipped_rows = process_file(
            csv_path=csv_path,
            window_size=args.window_size,
            step_size=args.step_size,
            verbose=args.verbose,
        )
        all_spectrum_tables.extend(spectra_tables)
        all_feature_rows.extend(feature_rows)
        all_skipped_rows.extend(skipped_rows)

    n_total_windows = len(all_feature_rows)
    metadata: dict[str, Any] = {
        "module_name": "Module 2: local spectral analysis",
        "script_name": script_path.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "window_size": int(args.window_size),
        "step_size": int(args.step_size),
        "n_input_files": int(len(input_files)),
        "n_segments_analyzed": int(len(all_spectrum_tables)),
        "n_segments_skipped": int(len(all_skipped_rows)),
        "n_total_windows": int(n_total_windows),
        "required_columns": REQUIRED_COLUMNS,
        "band_definitions": BAND_DEFINITIONS,
        "notes": [
            "signal = theta_signed in degrees",
            "x_axis = seq_index",
            "gaps handled by segmenting at has_gap_before == 1",
            "windows constrained to remain within contiguous observed segments",
            "each window mean-centered before FFT",
            "no smoothing",
            "no tapering/windowing",
            "no interpolation across gaps",
            "no plotting performed in this module",
            "outputs are spectrogram-ready numeric tables",
        ],
    }

    output_paths = write_outputs(
        output_dir=output_dir,
        spectrum_tables=all_spectrum_tables,
        feature_rows=all_feature_rows,
        skipped_rows=all_skipped_rows,
        metadata=metadata,
    )

    print(f"Total segments analyzed: {len(all_spectrum_tables)}")
    print(f"Total segments skipped: {len(all_skipped_rows)}")
    print(f"Total windows generated: {n_total_windows}")
    print(f"Main output: {output_paths['local_spectra_long']}")
    print(f"Features output: {output_paths['local_window_features']}")
    print(f"Skipped segments output: {output_paths['skipped_segments']}")
    print(f"Metadata output: {output_paths['run_metadata']}")


if __name__ == "__main__":
    main()
