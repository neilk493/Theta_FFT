"""Visualize precomputed global spectra for contiguous theta segments.

This script is intentionally separate from the analysis module by design. It
reads an existing global spectrum output table, produces one subplot per
segment with dominant-frequency markers, and does not perform any new spectral
analysis, preprocessing, or feature engineering.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS: list[str] = [
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
    "fft_bin",
    "frequency_cycles_per_residue",
    "period_residues",
    "amplitude",
    "power",
    "power_fraction_nonzero",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the plotting module."""

    parser = argparse.ArgumentParser(
        description="Plot precomputed global FFT power spectra with one subplot per segment."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("Theta_FFT/output/global_spectra/global_spectra_long.csv"),
        help="Path to the precomputed global_spectra_long.csv table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Theta_FFT/output/global_spectra_plots"),
        help="Directory where the combined figure will be written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG output resolution in dots per inch.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=12.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--row-height",
        type=float,
        default=2.6,
        help="Height in inches allocated to each subplot row.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-segment plotting progress.",
    )
    return parser.parse_args()


def ensure_output_dir(output_dir: Path) -> None:
    """Create the output directory if it does not already exist."""

    output_dir.mkdir(parents=True, exist_ok=True)


def resolve_cli_path(path_value: Path, script_dir: Path) -> Path:
    """Resolve a CLI path robustly from either the project root or its parent."""

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


def load_spectra_table(input_csv: Path) -> pd.DataFrame:
    """Load the precomputed global spectra table from disk."""

    if not input_csv.exists():
        raise FileNotFoundError(f"Input spectra CSV not found: {input_csv}")
    return pd.read_csv(input_csv)


def validate_spectra_table(df: pd.DataFrame, source_name: str) -> None:
    """Validate the plotting table schema and basic content."""

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{source_name}: missing required columns: {missing}")

    if df.empty:
        raise ValueError(f"{source_name}: input CSV is empty")


def get_segment_order(df: pd.DataFrame) -> list[str]:
    """Return a stable segment ordering by protein, chain, start index, and label."""

    metadata = (
        df.loc[:, ["segment_label", "protein", "chain", "segment_start_seq_index"]]
        .drop_duplicates()
        .sort_values(
            by=["protein", "chain", "segment_start_seq_index", "segment_label"],
            kind="mergesort",
        )
    )
    return metadata["segment_label"].astype(str).tolist()


def compute_dominant_frequency(segment_df: pd.DataFrame) -> float | None:
    """Return the dominant non-DC frequency for a segment, if one exists."""

    non_dc = segment_df.loc[segment_df["frequency_cycles_per_residue"] > 0].copy()
    if non_dc.empty:
        return None

    non_dc = non_dc.sort_values(
        by=["power", "frequency_cycles_per_residue"],
        ascending=[False, True],
        kind="mergesort",
    )
    dominant_row = non_dc.iloc[0]
    dominant_frequency = dominant_row["frequency_cycles_per_residue"]
    if pd.isna(dominant_frequency):
        return None
    return float(dominant_frequency)


def _format_segment_title(segment_df: pd.DataFrame) -> str:
    """Build the subplot title for one segment."""

    first_row = segment_df.iloc[0]
    protein = str(first_row["protein"])
    chain = str(first_row["chain"])
    segment_id = str(first_row["segment_id"])
    segment_length = int(first_row["segment_length"])
    return f"{protein}  |  chain {chain}  |  {segment_id}  |  n={segment_length} residues"


def _iter_segment_frames(df: pd.DataFrame, segment_order: Sequence[str]) -> list[tuple[str, pd.DataFrame]]:
    """Prepare sorted per-segment data frames in plotting order."""

    segment_frames: list[tuple[str, pd.DataFrame]] = []
    for segment_label in segment_order:
        segment_df = df.loc[df["segment_label"] == segment_label].copy()
        segment_df = segment_df.sort_values(
            by=["frequency_cycles_per_residue", "fft_bin"],
            kind="mergesort",
        )
        segment_frames.append((segment_label, segment_df))
    return segment_frames


def make_master_figure(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    fig_width: float,
    row_height: float,
    verbose: bool = False,
) -> list[tuple[str, Path, Path]]:
    """Create and save one stacked spectrum figure per protein."""

    ensure_output_dir(output_dir)

    segment_order = get_segment_order(df)
    n_segments = len(segment_order)
    if n_segments == 0:
        raise ValueError("No segments available to plot")

    protein_order = (
        df.loc[:, ["protein"]]
        .drop_duplicates()
        .sort_values(by=["protein"], kind="mergesort")
    )

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#1f1f1f",
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "lines.linewidth": 1.8,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    segment_frames = _iter_segment_frames(df, segment_order)
    written_paths: list[tuple[str, Path, Path]] = []

    for protein in protein_order["protein"].astype(str).tolist():
        protein_frames = [
            (segment_label, segment_df)
            for segment_label, segment_df in segment_frames
            if str(segment_df.iloc[0]["protein"]) == protein
        ]
        protein_frames.sort(
            key=lambda item: (
                int(item[1].iloc[0]["segment_start_seq_index"]),
                str(item[1].iloc[0]["chain"]),
                str(item[0]),
            )
        )

        n_protein_segments = len(protein_frames)
        figure_height = max(4.0, row_height * n_protein_segments)
        fig, axes = plt.subplots(
            n_protein_segments,
            1,
            figsize=(fig_width, figure_height),
            sharex=True,
        )

        if n_protein_segments == 1:
            axes_list = [axes]
        else:
            axes_list = list(axes)

        for index, ((segment_label, segment_df), ax) in enumerate(zip(protein_frames, axes_list), start=1):
            if verbose:
                print(f"Plotting {protein} segment {index}/{n_protein_segments}: {segment_label}")

            dominant_frequency = compute_dominant_frequency(segment_df)
            x_values = pd.to_numeric(segment_df["frequency_cycles_per_residue"], errors="coerce")
            y_values = pd.to_numeric(segment_df["power"], errors="coerce")

            ax.plot(
                x_values,
                y_values,
                color="#2C7BB6",
                linewidth=2.0,
                alpha=0.9,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
            ax.fill_between(x_values, y_values, alpha=0.15, color="#2C7BB6")
            ax.set_ylabel("Power", fontsize=11)
            ax.set_title(
                _format_segment_title(segment_df),
                loc="left",
                pad=8,
                fontsize=11,
                fontweight="bold",
            )
            ax.minorticks_on()
            ax.grid(True, axis="both")
            ax.grid(which="minor", alpha=0.3)
            ax.set_xlim(0.0, 0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.margins(x=0.0)

            if dominant_frequency is not None:
                ax.axvline(
                    dominant_frequency,
                    color="#D7191C",
                    linewidth=1.5,
                    linestyle="--",
                )
                y_max = float(y_values.max()) if not y_values.empty else 0.0
                annotation_y = y_max * 0.92 if y_max > 0 else 0.0
                annotation_x = min(dominant_frequency + 0.006, 0.498)
                ax.text(
                    annotation_x,
                    annotation_y,
                    f"{dominant_frequency:.4f}",
                    rotation=90,
                    fontsize=8,
                    color="#D7191C",
                    va="top",
                    ha="left",
                )

        axes_list[-1].set_xlabel("Frequency (cycles per residue)", fontsize=11)
        plt.tight_layout(h_pad=1.5)

        png_path = output_dir / f"{protein}_global_spectrum.png"
        pdf_path = output_dir / f"{protein}_global_spectrum.pdf"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Wrote PNG: {png_path}")
        print(f"Wrote PDF: {pdf_path}")
        written_paths.append((protein, png_path, pdf_path))

    return written_paths


def main() -> None:
    """Run the standalone global spectra plotting workflow."""

    args = parse_args()
    if args.dpi < 1:
        raise ValueError("--dpi must be at least 1")
    if args.fig_width <= 0:
        raise ValueError("--fig-width must be greater than 0")
    if args.row_height <= 0:
        raise ValueError("--row-height must be greater than 0")

    script_dir = Path(__file__).resolve().parent
    input_csv = resolve_cli_path(args.input_csv, script_dir)
    output_dir = resolve_cli_path(args.output_dir, script_dir)

    print(f"Input file: {input_csv}")
    df = load_spectra_table(input_csv)
    validate_spectra_table(df, input_csv.name)

    proteins_found = (
        df["protein"].astype(str).drop_duplicates().sort_values(kind="mergesort").tolist()
    )
    print(f"Proteins found: {len(proteins_found)}")

    make_master_figure(
        df=df,
        output_dir=output_dir,
        dpi=args.dpi,
        fig_width=args.fig_width,
        row_height=args.row_height,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
