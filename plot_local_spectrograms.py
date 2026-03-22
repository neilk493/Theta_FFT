"""Visualize precomputed sliding-window local spectra as spectrogram heatmaps.

This script is intentionally separate from the analysis module by design. It
creates one spectrogram per segment plus a stacked summary figure from existing
local spectral outputs, and it does not perform new FFT computation or
biological analysis.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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
    "fft_bin",
    "frequency_cycles_per_residue",
    "period_residues",
    "amplitude",
    "power",
    "power_fraction_nonzero",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for local spectrogram plotting."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot spectrogram heatmaps from precomputed sliding-window local "
            "spectral outputs."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("Theta_FFT/output/spectrograms/local_spectra_long.csv"),
        help="Path to the precomputed local_spectra_long.csv table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Theta_FFT/output/spectrogram_plots"),
        help="Directory where spectrogram figures will be written.",
    )
    parser.add_argument(
        "--color-value",
        choices=["power_fraction_nonzero", "power"],
        default="power_fraction_nonzero",
        help="Column to use for spectrogram color intensity.",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib colormap name for the heatmaps.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG output resolution in dots per inch.",
    )
    parser.add_argument(
        "--single-fig-width",
        type=float,
        default=10.0,
        help="Width in inches for individual segment spectrograms.",
    )
    parser.add_argument(
        "--single-fig-height",
        type=float,
        default=4.8,
        help="Height in inches for individual segment spectrograms.",
    )
    parser.add_argument(
        "--summary-fig-width",
        type=float,
        default=12.0,
        help="Width in inches for the stacked summary figure.",
    )
    parser.add_argument(
        "--summary-row-height",
        type=float,
        default=2.8,
        help="Height in inches allocated to each summary subplot row.",
    )
    parser.add_argument(
        "--drop-dc",
        action="store_true",
        help="Exclude the DC row (frequency == 0) from plots.",
    )
    parser.add_argument(
        "--keep-dc",
        action="store_true",
        help="Keep the DC row (frequency == 0) in plots.",
    )
    parser.add_argument(
        "--shared-scale",
        action="store_true",
        help="Use one shared color scale for all individual segment figures.",
    )
    parser.add_argument(
        "--per-segment-scale",
        action="store_true",
        help="Use a separate color scale for each individual segment figure.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information while plotting.",
    )
    return parser.parse_args()


def ensure_output_dir(output_dir: Path) -> None:
    """Create the output directory if it does not already exist."""

    output_dir.mkdir(parents=True, exist_ok=True)


def resolve_cli_path(path_value: Path, script_dir: Path) -> Path:
    """Resolve a CLI path robustly from inside the project or its parent."""

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


def load_local_spectra_table(input_csv: Path) -> pd.DataFrame:
    """Load the precomputed local spectra table from disk."""

    if not input_csv.exists():
        raise FileNotFoundError(f"Input local spectra CSV not found: {input_csv}")
    return pd.read_csv(input_csv)


def validate_local_spectra_table(df: pd.DataFrame, source_name: str) -> None:
    """Validate the plotting table schema and basic content."""

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{source_name}: missing required columns: {missing}")

    if df.empty:
        raise ValueError(f"{source_name}: input CSV is empty")


def resolve_plot_options(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve plotting behavior flags and validate mutually exclusive options."""

    if args.drop_dc and args.keep_dc:
        raise ValueError("Cannot provide both --drop-dc and --keep-dc")
    if args.shared_scale and args.per_segment_scale:
        raise ValueError("Cannot provide both --shared-scale and --per-segment-scale")
    if args.dpi < 1:
        raise ValueError("--dpi must be at least 1")
    if args.single_fig_width <= 0 or args.single_fig_height <= 0:
        raise ValueError("--single-fig-width and --single-fig-height must be greater than 0")
    if args.summary_fig_width <= 0 or args.summary_row_height <= 0:
        raise ValueError("--summary-fig-width and --summary-row-height must be greater than 0")

    drop_dc = True
    if args.keep_dc:
        drop_dc = False
    if args.drop_dc:
        drop_dc = True

    individual_scale_mode = "per-segment"
    if args.shared_scale:
        individual_scale_mode = "shared"
    if args.per_segment_scale:
        individual_scale_mode = "per-segment"

    return {
        "drop_dc": drop_dc,
        "individual_scale_mode": individual_scale_mode,
        "summary_scale_mode": "shared",
    }


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


def prepare_segment_matrix(
    segment_df: pd.DataFrame,
    color_value: str,
    drop_dc: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Pivot one segment into x, y, and z arrays for plotting."""

    if segment_df.empty:
        raise ValueError("Cannot prepare a matrix for an empty segment")

    working_df = segment_df.copy()
    numeric_columns = [
        "window_center_seq_index",
        "frequency_cycles_per_residue",
        color_value,
        "segment_length",
        "window_length",
    ]
    for column in numeric_columns:
        working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

    working_df = working_df.dropna(
        subset=["window_center_seq_index", "frequency_cycles_per_residue", color_value]
    )

    if drop_dc:
        working_df = working_df.loc[working_df["frequency_cycles_per_residue"] != 0].copy()

    if working_df.empty:
        raise ValueError(
            f"No rows remain for segment {segment_df.iloc[0]['segment_label']} after filtering"
        )

    pivot = working_df.pivot_table(
        index="frequency_cycles_per_residue",
        columns="window_center_seq_index",
        values=color_value,
        aggfunc="first",
        sort=True,
    )
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    if pivot.empty:
        raise ValueError(
            f"Malformed pivot for segment {segment_df.iloc[0]['segment_label']}: empty matrix"
        )

    if pivot.index.has_duplicates or pivot.columns.has_duplicates:
        raise ValueError(
            f"Malformed pivot for segment {segment_df.iloc[0]['segment_label']}: duplicate axes"
        )

    z_matrix = pivot.to_numpy(dtype=float)
    if z_matrix.ndim != 2 or z_matrix.shape[0] == 0 or z_matrix.shape[1] == 0:
        raise ValueError(
            f"Malformed pivot for segment {segment_df.iloc[0]['segment_label']}: invalid dimensions"
        )

    x_values = pivot.columns.to_numpy(dtype=float)
    y_values = pivot.index.to_numpy(dtype=float)

    first_row = working_df.iloc[0]
    metadata = {
        "protein": str(first_row["protein"]),
        "chain": str(first_row["chain"]),
        "segment_id": str(first_row["segment_id"]),
        "segment_label": str(first_row["segment_label"]),
        "segment_length": int(round(float(first_row["segment_length"]))),
        "n_windows": int(len(x_values)),
        "window_length": int(round(float(first_row["window_length"]))),
    }
    return x_values, y_values, z_matrix, metadata


def compute_color_limits(values: np.ndarray, percentile_max: float = 99.0) -> tuple[float, float]:
    """Compute robust color limits from finite plotted values."""

    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("Cannot compute color limits from an empty value array")

    vmin = 0.0
    vmax = float(np.percentile(finite_values, percentile_max))
    if not np.isfinite(vmax):
        raise ValueError("Computed vmax is not finite")
    if vmax <= vmin:
        vmax = float(finite_values.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _configure_matplotlib_style() -> None:
    """Apply clean, publication-oriented matplotlib styling."""

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "#222222",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.35,
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "image.interpolation": "nearest",
        }
    )


def _compute_edges(values: np.ndarray) -> np.ndarray:
    """Convert sorted center coordinates into pcolormesh cell edges."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Cannot compute edges for an empty coordinate array")
    if values.size == 1:
        step = 1.0
        return np.array([values[0] - 0.5 * step, values[0] + 0.5 * step], dtype=float)

    diffs = np.diff(values)
    if np.any(diffs <= 0):
        raise ValueError("Coordinate values must be strictly increasing to compute cell edges")

    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = values[:-1] + diffs / 2.0
    edges[0] = values[0] - diffs[0] / 2.0
    edges[-1] = values[-1] + diffs[-1] / 2.0
    return edges


def _colorbar_label(color_value: str) -> str:
    """Return the human-readable colorbar label."""

    if color_value == "power_fraction_nonzero":
        return "Local spectral power fraction"
    if color_value == "power":
        return "Local spectral power"
    return color_value


def _single_title(metadata: dict[str, Any]) -> str:
    """Build a clean title for one segment spectrogram."""

    return (
        f"{metadata['protein']} | chain {metadata['chain']} | {metadata['segment_id']} | "
        f"segment n={metadata['segment_length']} | windows={metadata['n_windows']} | "
        f"window size={metadata['window_length']}"
    )


def plot_single_segment_spectrogram(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_matrix: np.ndarray,
    metadata: dict[str, Any],
    output_dir: Path,
    color_value: str,
    cmap: str,
    dpi: int,
    fig_width: float,
    fig_height: float,
    vmin: float,
    vmax: float,
    verbose: bool = False,
) -> tuple[Path, Path]:
    """Plot and save one spectrogram figure for a single segment."""

    x_edges = _compute_edges(x_values)
    y_edges = _compute_edges(y_values)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        z_matrix,
        cmap=cmap,
        shading="flat",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(_single_title(metadata), loc="left", pad=10, fontweight="bold")
    ax.set_xlabel("Residue position (window center, seq_index)")
    ax.set_ylabel("Frequency (cycles per residue)")
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(_colorbar_label(color_value))
    cbar.outline.set_linewidth(0.7)

    segment_label = metadata["segment_label"]
    png_path = output_dir / f"{segment_label}_spectrogram.png"
    pdf_path = output_dir / f"{segment_label}_spectrogram.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"Saved PNG: {png_path}")
        print(f"Saved PDF: {pdf_path}")

    return png_path, pdf_path


def plot_summary_spectrograms(
    segment_payloads: list[dict[str, Any]],
    output_dir: Path,
    color_value: str,
    cmap: str,
    dpi: int,
    fig_width: float,
    row_height: float,
    vmin: float,
    vmax: float,
    verbose: bool = False,
) -> tuple[Path, Path]:
    """Plot and save the vertically stacked all-segments summary figure."""

    n_segments = len(segment_payloads)
    if n_segments == 0:
        raise ValueError("No segments available to plot in the summary figure")

    figure_height = max(4.0, row_height * n_segments)
    fig, axes = plt.subplots(
        n_segments,
        1,
        figsize=(fig_width, figure_height),
        constrained_layout=True,
    )

    axes_list = [axes] if n_segments == 1 else list(axes)
    mesh = None
    for index, (payload, ax) in enumerate(zip(segment_payloads, axes_list), start=1):
        x_edges = _compute_edges(payload["x_values"])
        y_edges = _compute_edges(payload["y_values"])
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            payload["z_matrix"],
            cmap=cmap,
            shading="flat",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(_single_title(payload["metadata"]), loc="left", pad=8, fontsize=10.5)
        ax.set_ylabel("Frequency (cycles per residue)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        if index == n_segments:
            ax.set_xlabel("Residue position (window center, seq_index)")
        else:
            ax.tick_params(labelbottom=False)

        if verbose:
            print(
                f"Summary subplot {index}/{n_segments}: "
                f"{payload['metadata']['segment_label']}"
            )

    assert mesh is not None
    cbar = fig.colorbar(mesh, ax=axes_list, pad=0.01, shrink=0.98)
    cbar.set_label(_colorbar_label(color_value))
    cbar.outline.set_linewidth(0.7)

    png_path = output_dir / "all_segments_spectrograms.png"
    pdf_path = output_dir / "all_segments_spectrograms.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"Saved summary PNG: {png_path}")
        print(f"Saved summary PDF: {pdf_path}")

    return png_path, pdf_path


def main() -> None:
    """Run the standalone local spectrogram plotting workflow."""

    args = parse_args()
    options = resolve_plot_options(args)
    _configure_matplotlib_style()

    script_dir = Path(__file__).resolve().parent
    input_csv = resolve_cli_path(args.input_csv, script_dir)
    output_dir = resolve_cli_path(args.output_dir, script_dir)
    ensure_output_dir(output_dir)

    print(f"Input file: {input_csv}")
    df = load_local_spectra_table(input_csv)
    validate_local_spectra_table(df, input_csv.name)

    segment_order = get_segment_order(df)
    print(f"Unique segments plotted: {len(segment_order)}")
    if not segment_order:
        raise ValueError("No segments available to plot")

    segment_payloads: list[dict[str, Any]] = []
    all_summary_values: list[np.ndarray] = []

    for index, segment_label in enumerate(segment_order, start=1):
        segment_df = df.loc[df["segment_label"].astype(str) == segment_label].copy()
        x_values, y_values, z_matrix, metadata = prepare_segment_matrix(
            segment_df=segment_df,
            color_value=args.color_value,
            drop_dc=options["drop_dc"],
        )
        segment_payloads.append(
            {
                "x_values": x_values,
                "y_values": y_values,
                "z_matrix": z_matrix,
                "metadata": metadata,
            }
        )
        all_summary_values.append(z_matrix[np.isfinite(z_matrix)])
        if args.verbose:
            print(f"Prepared segment {index}/{len(segment_order)}: {segment_label}")

    shared_summary_values = np.concatenate(all_summary_values) if all_summary_values else np.array([])
    summary_vmin, summary_vmax = compute_color_limits(shared_summary_values)

    if options["individual_scale_mode"] == "shared":
        single_vmin, single_vmax = summary_vmin, summary_vmax
    else:
        single_vmin = single_vmax = math.nan

    for index, payload in enumerate(segment_payloads, start=1):
        if args.verbose:
            print(
                f"Plotting segment {index}/{len(segment_payloads)}: "
                f"{payload['metadata']['segment_label']}"
            )

        if options["individual_scale_mode"] == "shared":
            vmin, vmax = single_vmin, single_vmax
        else:
            vmin, vmax = compute_color_limits(payload["z_matrix"])

        plot_single_segment_spectrogram(
            x_values=payload["x_values"],
            y_values=payload["y_values"],
            z_matrix=payload["z_matrix"],
            metadata=payload["metadata"],
            output_dir=output_dir,
            color_value=args.color_value,
            cmap=args.cmap,
            dpi=args.dpi,
            fig_width=args.single_fig_width,
            fig_height=args.single_fig_height,
            vmin=vmin,
            vmax=vmax,
            verbose=args.verbose,
        )

    plot_summary_spectrograms(
        segment_payloads=segment_payloads,
        output_dir=output_dir,
        color_value=args.color_value,
        cmap=args.cmap,
        dpi=args.dpi,
        fig_width=args.summary_fig_width,
        row_height=args.summary_row_height,
        vmin=summary_vmin,
        vmax=summary_vmax,
        verbose=args.verbose,
    )

    print(f"Output images written to: {output_dir}")


if __name__ == "__main__":
    main()
