"""Visualize precomputed window-level comparison outputs as 3D t-SNE embeddings.

This script is intentionally separate from analysis. It reads previously generated
comparison tables, creates 3D t-SNE views colored by protein and motif cluster,
and does not recompute FFT features, similarity values, or motif clustering.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

FIG_DPI = 300
WINDOW_FEATURES_FILE = "window_features_standardized.csv"
MOTIF_CLUSTER_ASSIGNMENTS_FILE = "motif_cluster_assignments.csv"
REQUIRED_INPUT_FILES = {
    "window_features": WINDOW_FEATURES_FILE,
    "motif_cluster_assignments": MOTIF_CLUSTER_ASSIGNMENTS_FILE,
}
WINDOW_FEATURES_REQUIRED_COLUMNS = {
    "protein",
    "chain",
    "segment_label",
    "window_label",
    "window_center_seq_index",
}
MOTIF_ASSIGNMENTS_REQUIRED_COLUMNS = {
    "window_label",
    "protein",
    "chain",
    "motif_cluster_id",
    "distance_to_centroid",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for 3D t-SNE comparison plotting."""

    parser = argparse.ArgumentParser(
        description=(
            "Create 3D t-SNE visualizations from precomputed standardized "
            "window-level comparison features."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Theta_FFT/output/comparison"),
        help="Directory containing precomputed comparison tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Theta_FFT/output/comparison_plots"),
        help="Directory where 3D t-SNE plots will be written.",
    )
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument(
        "--learning-rate", type=float, default=200.0, help="t-SNE learning rate."
    )
    parser.add_argument(
        "--n-iter", type=int, default=1000, help="Number of t-SNE optimization iterations."
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for t-SNE initialization."
    )
    parser.add_argument(
        "--view-elev",
        type=float,
        default=22.0,
        help="Elevation angle for the default 3D view.",
    )
    parser.add_argument(
        "--view-azim",
        type=float,
        default=38.0,
        help="Azimuth angle for the default 3D view.",
    )
    parser.add_argument(
        "--alt-view-elev",
        type=float,
        default=14.0,
        help="Elevation angle for the alternate 3D view.",
    )
    parser.add_argument(
        "--alt-view-azim",
        type=float,
        default=118.0,
        help="Azimuth angle for the alternate 3D view.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print extra progress information.")
    return parser.parse_args()


def ensure_output_dir(output_dir: Path) -> None:
    """Create the output directory if it does not already exist."""

    output_dir.mkdir(parents=True, exist_ok=True)


def resolve_cli_path(path_value: Path, script_dir: Path) -> Path:
    """Resolve CLI paths robustly relative to the script location."""

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


def load_inputs(input_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the required comparison tables from disk."""

    data: dict[str, pd.DataFrame] = {}
    for key, filename in REQUIRED_INPUT_FILES.items():
        file_path = input_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Required input file not found: {file_path}")
        data[key] = pd.read_csv(file_path)
        print(f"Loaded input: {file_path}")
    return data


def validate_inputs(data: dict[str, pd.DataFrame]) -> None:
    """Validate the required plotting inputs and their schemas."""

    missing_keys = [key for key in REQUIRED_INPUT_FILES if key not in data]
    if missing_keys:
        raise ValueError(f"Missing loaded input tables: {', '.join(missing_keys)}")

    window_df = data["window_features"]
    motif_df = data["motif_cluster_assignments"]

    if window_df.empty:
        raise ValueError(f"{WINDOW_FEATURES_FILE}: input table is empty")
    if motif_df.empty:
        raise ValueError(f"{MOTIF_CLUSTER_ASSIGNMENTS_FILE}: input table is empty")

    missing_window_columns = [
        column for column in WINDOW_FEATURES_REQUIRED_COLUMNS if column not in window_df.columns
    ]
    if missing_window_columns:
        raise ValueError(
            f"{WINDOW_FEATURES_FILE}: missing required columns: "
            f"{', '.join(sorted(missing_window_columns))}"
        )

    z_columns = [column for column in window_df.columns if column.startswith("z__")]
    if not z_columns:
        raise ValueError(f"{WINDOW_FEATURES_FILE}: no feature columns with prefix 'z__' were found")

    missing_motif_columns = [
        column for column in MOTIF_ASSIGNMENTS_REQUIRED_COLUMNS if column not in motif_df.columns
    ]
    if missing_motif_columns:
        raise ValueError(
            f"{MOTIF_CLUSTER_ASSIGNMENTS_FILE}: missing required columns: "
            f"{', '.join(sorted(missing_motif_columns))}"
        )

    for protein in window_df["protein"].astype(str).unique():
        get_protein_class(protein)


def get_protein_class(protein: str) -> str:
    """Return the structural class label for a protein."""

    helix = {"2HHB", "1GZM"}
    sheet = {"1TEN", "1FNA", "2IGF"}
    mixed = {"1UBQ", "1PKK", "1LYZ", "2PTN"}
    if protein in helix:
        return "helix"
    if protein in sheet:
        return "sheet"
    if protein in mixed:
        return "mixed"
    raise ValueError(f"Unknown protein for structural class mapping: {protein}")


def get_protein_color_map() -> dict[str, str]:
    """Return the fixed protein color palette."""

    return {
        "2HHB": "#C0392B",
        "1GZM": "#E74C3C",
        "1TEN": "#2980B9",
        "1FNA": "#5DADE2",
        "2IGF": "#1A5276",
        "1UBQ": "#7D6608",
        "1PKK": "#148F77",
        "1LYZ": "#82E0AA",
        "2PTN": "#7F8C8D",
    }


def get_class_accent_colors() -> dict[str, str]:
    """Return the structural class accent colors."""

    return {
        "helix": "#D35400",
        "sheet": "#2471A3",
        "mixed": "#6E7B34",
    }


def get_protein_order() -> list[str]:
    """Return the fixed plotting order for proteins."""

    return ["2HHB", "1GZM", "1TEN", "1FNA", "2IGF", "1PKK", "1UBQ", "1LYZ", "2PTN"]


def get_motif_color_map(cluster_ids: list[str]) -> dict[str, str]:
    """Return a stable motif cluster palette."""

    sorted_cluster_ids = sorted({str(cluster_id) for cluster_id in cluster_ids}, key=cluster_sort_key)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(index % cmap.N) for index in range(len(sorted_cluster_ids))]
    return {cluster_id: colors[index] for index, cluster_id in enumerate(sorted_cluster_ids)}


def cluster_sort_key(cluster_id: str) -> tuple[int, float, str]:
    """Sort motif cluster identifiers numerically when possible."""

    text = str(cluster_id)
    try:
        return (0, float(text), text)
    except ValueError:
        return (1, float("inf"), text)


def configure_matplotlib_style() -> None:
    """Apply a clean, publication-oriented plotting style."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.edgecolor": "#B5BCC4",
            "axes.linewidth": 0.9,
            "axes.labelcolor": "#222222",
            "xtick.color": "#404040",
            "ytick.color": "#404040",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.color": "#DCE2E8",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#D8DDE3",
            "legend.framealpha": 0.98,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def compute_pairwise_squared_distances(X: np.ndarray) -> np.ndarray:
    """Compute the full pairwise squared Euclidean distance matrix."""

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    squared_norms = np.sum(X * X, axis=1, keepdims=True)
    distances = squared_norms + squared_norms.T - 2.0 * (X @ X.T)
    np.maximum(distances, 0.0, out=distances)
    return distances


def binary_search_sigma(
    dist_row: np.ndarray,
    target_perplexity: float,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> tuple[np.ndarray, float]:
    """Find a conditional probability row matching the target perplexity."""

    if target_perplexity <= 0:
        raise ValueError("target_perplexity must be greater than 0")
    if dist_row.ndim != 1:
        raise ValueError("dist_row must be one-dimensional")
    if dist_row.size == 0:
        raise ValueError("dist_row must contain at least one distance")

    log_target = np.log(target_perplexity)
    beta = 1.0
    beta_min = -np.inf
    beta_max = np.inf
    probabilities = np.full(dist_row.shape, 1.0 / dist_row.size, dtype=float)

    for _ in range(max_iter):
        scaled = -dist_row * beta
        scaled -= scaled.max()
        probabilities = np.exp(scaled)
        probabilities_sum = probabilities.sum()
        if probabilities_sum <= 0.0 or not np.isfinite(probabilities_sum):
            probabilities = np.full(dist_row.shape, 1.0 / dist_row.size, dtype=float)
            probabilities_sum = 1.0
        probabilities /= probabilities_sum
        probabilities = np.maximum(probabilities, 1e-12)
        probabilities /= probabilities.sum()

        entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))
        entropy_diff = entropy - log_target
        if abs(entropy_diff) < tol:
            break

        if entropy_diff > 0:
            beta_min = beta
            beta = beta * 2.0 if not np.isfinite(beta_max) else 0.5 * (beta + beta_max)
        else:
            beta_max = beta
            beta = beta / 2.0 if not np.isfinite(beta_min) else 0.5 * (beta + beta_min)

    sigma = np.sqrt(1.0 / max(2.0 * beta, 1e-12))
    return probabilities, float(sigma)


def compute_joint_probabilities(X: np.ndarray, perplexity: float, verbose: bool = False) -> np.ndarray:
    """Compute the symmetric joint probability matrix P for t-SNE."""

    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("t-SNE requires at least two samples")
    if perplexity >= n_samples:
        raise ValueError(
            f"perplexity must be smaller than the number of samples; got {perplexity} for {n_samples} samples"
        )

    distances = compute_pairwise_squared_distances(X)
    conditional = np.zeros((n_samples, n_samples), dtype=float)

    for i in range(n_samples):
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        row_probabilities, sigma = binary_search_sigma(distances[i, mask], perplexity)
        conditional[i, mask] = row_probabilities
        if verbose and (i + 1) % 100 == 0:
            print(f"Computed conditional probabilities for {i + 1}/{n_samples} points (sigma={sigma:.4f})")

    joint = conditional + conditional.T
    np.fill_diagonal(joint, 0.0)
    joint_sum = joint.sum()
    if joint_sum <= 0.0 or not np.isfinite(joint_sum):
        raise ValueError("Failed to compute a valid joint probability matrix for t-SNE")
    joint /= joint_sum
    joint = np.maximum(joint, 1e-12)
    np.fill_diagonal(joint, 0.0)
    joint /= np.maximum(joint.sum(), 1e-12)
    return joint


def run_tsne_3d(
    X: np.ndarray,
    perplexity: float,
    learning_rate: float,
    n_iter: int,
    random_seed: int,
    verbose: bool = False,
) -> np.ndarray:
    """Run exact 3D t-SNE from scratch using NumPy only."""

    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("t-SNE requires at least two rows")

    early_exaggeration = 12.0
    early_exaggeration_iters = 250
    initial_momentum = 0.5
    later_momentum = 0.8
    min_gain = 0.01

    P = compute_joint_probabilities(X, perplexity, verbose=verbose)
    P *= early_exaggeration
    P = np.maximum(P, 1e-12)
    np.fill_diagonal(P, 0.0)
    P /= np.maximum(P.sum(), 1e-12)

    rng = np.random.default_rng(random_seed)
    Y = rng.normal(loc=0.0, scale=1e-4, size=(n_samples, 3))
    Y_incs = np.zeros_like(Y)
    gains = np.ones_like(Y)

    for iteration in range(n_iter):
        sum_Y = np.sum(Y * Y, axis=1)
        squared_distances = -2.0 * (Y @ Y.T) + sum_Y[:, None] + sum_Y[None, :]
        num = 1.0 / (1.0 + squared_distances)
        np.fill_diagonal(num, 0.0)
        Q = num / np.maximum(num.sum(), 1e-12)
        Q = np.maximum(Q, 1e-12)
        np.fill_diagonal(Q, 0.0)
        Q /= np.maximum(Q.sum(), 1e-12)

        pq_diff = P - Q
        weighted = 4.0 * pq_diff * num
        row_sums = np.sum(weighted, axis=1)
        dY = Y * row_sums[:, None] - weighted @ Y

        sign_changes = np.sign(dY) != np.sign(Y_incs)
        gains = np.where(sign_changes, gains + 0.2, gains * 0.8)
        gains = np.maximum(gains, min_gain)

        momentum = initial_momentum if iteration < 20 else later_momentum
        Y_incs = momentum * Y_incs - learning_rate * gains * dY
        Y += Y_incs
        Y -= np.mean(Y, axis=0, keepdims=True)

        if iteration + 1 == early_exaggeration_iters:
            P /= early_exaggeration
            P = np.maximum(P, 1e-12)
            np.fill_diagonal(P, 0.0)
            P /= np.maximum(P.sum(), 1e-12)

        if verbose and ((iteration + 1) % 100 == 0 or iteration == 0 or iteration + 1 == n_iter):
            kl = float(np.sum(P * np.log(np.maximum(P, 1e-12) / np.maximum(Q, 1e-12))))
            print(f"t-SNE iteration {iteration + 1}/{n_iter}: KL={kl:.6f}")

    return Y


def build_tsne_3d_dataframe(
    features_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    coords: np.ndarray,
) -> pd.DataFrame:
    """Build the final plotting dataframe with merged motif assignments."""

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be a 2D array with exactly three columns")
    if len(features_df) != coords.shape[0]:
        raise ValueError(
            "The number of coordinate rows must match the number of plotted feature rows"
        )

    tsne_df = features_df.loc[:, ["window_label", "protein", "chain", "segment_label"]].copy()
    for column in ["window_label", "protein", "chain", "segment_label"]:
        tsne_df[column] = tsne_df[column].astype(str)

    tsne_df["tsne_1"] = coords[:, 0]
    tsne_df["tsne_2"] = coords[:, 1]
    tsne_df["tsne_3"] = coords[:, 2]

    motif_merge = assignments_df.loc[:, ["window_label", "motif_cluster_id"]].copy()
    motif_merge["window_label"] = motif_merge["window_label"].astype(str)

    duplicate_windows = motif_merge["window_label"].duplicated()
    if duplicate_windows.any():
        duplicates = motif_merge.loc[duplicate_windows, "window_label"].astype(str).head(5).tolist()
        raise ValueError(
            f"{MOTIF_CLUSTER_ASSIGNMENTS_FILE}: expected one row per plotted window_label; duplicate examples: "
            f"{', '.join(duplicates)}"
        )

    merged = tsne_df.merge(
        motif_merge,
        on="window_label",
        how="left",
        validate="one_to_one",
    )
    if merged["motif_cluster_id"].isna().any():
        missing_count = int(merged["motif_cluster_id"].isna().sum())
        raise ValueError(
            f"{MOTIF_CLUSTER_ASSIGNMENTS_FILE}: motif_cluster_id missing for {missing_count} plotted windows after merge"
        )

    merged["motif_cluster_id"] = merged["motif_cluster_id"].astype(str)
    return merged


def save_tsne_3d_coordinates(tsne_df: pd.DataFrame, output_dir: Path) -> Path:
    """Save the 3D t-SNE coordinates for later reuse."""

    output_path = output_dir / "tsne_3d_coordinates.csv"
    columns = [
        "window_label",
        "protein",
        "chain",
        "segment_label",
        "motif_cluster_id",
        "tsne_1",
        "tsne_2",
        "tsne_3",
    ]
    tsne_df.loc[:, columns].to_csv(output_path, index=False)
    print(f"Saved coordinates: {output_path}")
    return output_path


def style_3d_axes(ax: Any) -> None:
    """Apply clean, restrained styling to a 3D axis."""

    ax.set_facecolor("white")
    ax.tick_params(colors="#4B5661", labelsize=9, pad=3)
    ax.xaxis.label.set_color("#222222")
    ax.yaxis.label.set_color("#222222")
    ax.zaxis.label.set_color("#222222")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color("#C8CED6")
        axis.line.set_linewidth(0.8)
        if hasattr(axis, "pane"):
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.06))
            axis.pane.set_edgecolor((0.82, 0.85, 0.88, 0.55))
        if hasattr(axis, "_axinfo"):
            axis._axinfo["grid"]["color"] = (0.86, 0.89, 0.92, 0.7)
            axis._axinfo["grid"]["linewidth"] = 0.6
            axis._axinfo["grid"]["linestyle"] = "-"

    ax.grid(True)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def set_balanced_3d_limits(ax: Any, coords: np.ndarray) -> None:
    """Set visually balanced limits around the 3D embedding."""

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    centers = (mins + maxs) / 2.0
    spans = maxs - mins
    max_span = float(np.max(spans))
    half_range = max(max_span * 0.55, 1e-6)

    ax.set_xlim(centers[0] - half_range, centers[0] + half_range)
    ax.set_ylim(centers[1] - half_range, centers[1] + half_range)
    ax.set_zlim(centers[2] - half_range, centers[2] + half_range)


def create_figure_paths(output_dir: Path, stem: str, suffix: str = "") -> tuple[Path, Path]:
    """Return PNG and PDF output paths for a figure stem."""

    suffix_text = suffix if suffix else ""
    return (
        output_dir / f"{stem}{suffix_text}.png",
        output_dir / f"{stem}{suffix_text}.pdf",
    )


def save_figure(fig: plt.Figure, png_path: Path, pdf_path: Path) -> tuple[Path, Path]:
    """Save a figure to both PNG and PDF."""

    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    return png_path, pdf_path


def plot_tsne_3d_by_protein(
    tsne_df: pd.DataFrame,
    output_dir: Path,
    elev: float,
    azim: float,
    suffix: str = "",
) -> tuple[Path, Path]:
    """Plot the shared 3D t-SNE embedding colored by protein."""

    protein_colors = get_protein_color_map()
    protein_order = get_protein_order()
    accent_colors = get_class_accent_colors()
    coords = tsne_df.loc[:, ["tsne_1", "tsne_2", "tsne_3"]].to_numpy(dtype=float)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    for protein in protein_order:
        subset = tsne_df.loc[tsne_df["protein"] == protein]
        if subset.empty:
            continue
        ax.scatter(
            subset["tsne_1"],
            subset["tsne_2"],
            subset["tsne_3"],
            s=18,
            alpha=0.72,
            c=protein_colors[protein],
            edgecolors="white",
            linewidths=0.25,
            depthshade=False,
        )

    fig.suptitle("Theta-pp FFT Window Embedding (3D t-SNE)", fontsize=17, fontweight="bold", y=0.97)
    ax.set_title("colored by protein", fontsize=11, color="#56606B", pad=12)
    ax.set_xlabel("t-SNE dimension 1", labelpad=10)
    ax.set_ylabel("t-SNE dimension 2", labelpad=10)
    ax.set_zlabel("t-SNE dimension 3", labelpad=10)
    ax.view_init(elev=elev, azim=azim)
    style_3d_axes(ax)
    set_balanced_3d_limits(ax, coords)

    class_sections = [
        ("Helix", "helix", ["2HHB", "1GZM"]),
        ("Sheet", "sheet", ["1TEN", "1FNA", "2IGF"]),
        ("Mixed", "mixed", ["1PKK", "1UBQ", "1LYZ", "2PTN"]),
    ]
    handles: list[Line2D] = []
    labels: list[str] = []
    heading_indices: list[tuple[int, str]] = []
    for section_label, class_key, proteins in class_sections:
        handles.append(Line2D([], [], linestyle="none"))
        labels.append(section_label)
        heading_indices.append((len(labels) - 1, class_key))
        for protein in proteins:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=protein_colors[protein],
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    markersize=7.5,
                )
            )
            labels.append(protein)

    legend = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        handlelength=1.0,
        labelspacing=0.7,
        handletextpad=0.7,
    )
    for index, text in enumerate(legend.get_texts()):
        text.set_color("#2A2F35")
        for heading_index, class_key in heading_indices:
            if index == heading_index:
                text.set_fontweight("bold")
                text.set_color(accent_colors[class_key])

    fig.subplots_adjust(top=0.89, right=0.79)
    png_path, pdf_path = create_figure_paths(output_dir, "tsne_3d_by_protein", suffix=suffix)
    return save_figure(fig, png_path, pdf_path)


def plot_tsne_3d_by_motif_cluster(
    tsne_df: pd.DataFrame,
    output_dir: Path,
    elev: float,
    azim: float,
    suffix: str = "",
) -> tuple[Path, Path]:
    """Plot the shared 3D t-SNE embedding colored by motif cluster."""

    motif_cluster_ids = sorted(
        tsne_df["motif_cluster_id"].astype(str).unique().tolist(),
        key=cluster_sort_key,
    )
    cluster_colors = get_motif_color_map(motif_cluster_ids)
    coords = tsne_df.loc[:, ["tsne_1", "tsne_2", "tsne_3"]].to_numpy(dtype=float)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    for cluster_id in motif_cluster_ids:
        subset = tsne_df.loc[tsne_df["motif_cluster_id"].astype(str) == cluster_id]
        if subset.empty:
            continue
        ax.scatter(
            subset["tsne_1"],
            subset["tsne_2"],
            subset["tsne_3"],
            s=18,
            alpha=0.72,
            c=[cluster_colors[cluster_id]],
            edgecolors="white",
            linewidths=0.25,
            depthshade=False,
        )

    fig.suptitle("Theta-pp FFT Window Embedding (3D t-SNE)", fontsize=17, fontweight="bold", y=0.97)
    ax.set_title("colored by motif cluster", fontsize=11, color="#56606B", pad=12)
    ax.set_xlabel("t-SNE dimension 1", labelpad=10)
    ax.set_ylabel("t-SNE dimension 2", labelpad=10)
    ax.set_zlabel("t-SNE dimension 3", labelpad=10)
    ax.view_init(elev=elev, azim=azim)
    style_3d_axes(ax)
    set_balanced_3d_limits(ax, coords)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=cluster_colors[cluster_id],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=7.5,
            label=cluster_id,
        )
        for cluster_id in motif_cluster_ids
    ]
    n_clusters = len(motif_cluster_ids)
    legend_ncol = 1 if n_clusters <= 10 else 2
    ax.legend(
        handles=handles,
        labels=motif_cluster_ids,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        title="Motif cluster",
        title_fontsize=10,
        ncol=legend_ncol,
        handletextpad=0.7,
        labelspacing=0.7,
    )

    fig.subplots_adjust(top=0.89, right=0.78 if n_clusters <= 10 else 0.72)
    png_path, pdf_path = create_figure_paths(output_dir, "tsne_3d_by_motif_cluster", suffix=suffix)
    return save_figure(fig, png_path, pdf_path)


def main() -> None:
    """Run the standalone 3D t-SNE plotting workflow."""

    args = parse_args()
    if args.perplexity <= 0:
        raise ValueError("--perplexity must be greater than 0")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than 0")
    if args.n_iter < 250:
        raise ValueError("--n-iter must be at least 250")

    configure_matplotlib_style()

    script_dir = Path(__file__).resolve().parent
    input_dir = resolve_cli_path(args.input_dir, script_dir)
    output_dir = resolve_cli_path(args.output_dir, script_dir)
    ensure_output_dir(output_dir)

    data = load_inputs(input_dir)
    validate_inputs(data)

    window_df = data["window_features"].copy()
    motif_df = data["motif_cluster_assignments"].copy()

    z_columns = sorted([column for column in window_df.columns if column.startswith("z__")])
    if args.verbose:
        print(f"Using {len(z_columns)} standardized feature columns for 3D t-SNE")

    missing_feature_mask = window_df[z_columns].isna().any(axis=1)
    n_dropped = int(missing_feature_mask.sum())
    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows with missing z__ features before 3D t-SNE")
    tsne_input_df = window_df.loc[~missing_feature_mask].copy()

    if tsne_input_df.empty:
        raise ValueError("No rows remained for 3D t-SNE after dropping missing feature values")

    X = tsne_input_df[z_columns].to_numpy(dtype=float)
    print(f"Windows used for 3D t-SNE: {X.shape[0]}")

    tsne_coordinates = run_tsne_3d(
        X=X,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        random_seed=args.random_seed,
        verbose=args.verbose,
    )

    tsne_df = build_tsne_3d_dataframe(tsne_input_df, motif_df, tsne_coordinates)
    save_tsne_3d_coordinates(tsne_df, output_dir)

    if args.verbose:
        print("Plotting 3D t-SNE embedding colored by protein")
    plot_tsne_3d_by_protein(
        tsne_df=tsne_df,
        output_dir=output_dir,
        elev=args.view_elev,
        azim=args.view_azim,
    )

    if args.verbose:
        print("Plotting alternate 3D t-SNE protein view")
    plot_tsne_3d_by_protein(
        tsne_df=tsne_df,
        output_dir=output_dir,
        elev=args.alt_view_elev,
        azim=args.alt_view_azim,
        suffix="_alt_view",
    )

    if args.verbose:
        print("Plotting 3D t-SNE embedding colored by motif cluster")
    plot_tsne_3d_by_motif_cluster(
        tsne_df=tsne_df,
        output_dir=output_dir,
        elev=args.view_elev,
        azim=args.view_azim,
    )

    if args.verbose:
        print("Plotting alternate 3D t-SNE motif-cluster view")
    plot_tsne_3d_by_motif_cluster(
        tsne_df=tsne_df,
        output_dir=output_dir,
        elev=args.alt_view_elev,
        azim=args.alt_view_azim,
        suffix="_alt_view",
    )


if __name__ == "__main__":
    main()
