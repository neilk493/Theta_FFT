"""Visualize precomputed protein and motif comparison outputs.

This script is intentionally separate from analysis by design. It reads
already-generated comparison tables, creates t-SNE embeddings, a protein
similarity heatmap, a spectral distance-distribution figure, and a motif
cluster composition plot, and it does not recompute FFT, similarity, or
clustering.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


WINDOW_FEATURES_FILE = "window_features_standardized.csv"
PROTEIN_SIMILARITY_MATRIX_FILE = "protein_similarity_matrix.csv"
MOTIF_CLUSTER_ASSIGNMENTS_FILE = "motif_cluster_assignments.csv"
PROTEIN_SIMILARITY_LONG_FILE = "protein_similarity_long.csv"
MOTIF_CLUSTER_PROTEIN_COUNTS_FILE = "motif_cluster_protein_counts.csv"

REQUIRED_INPUT_FILES: dict[str, str] = {
    "window_features": WINDOW_FEATURES_FILE,
    "protein_similarity_matrix": PROTEIN_SIMILARITY_MATRIX_FILE,
    "motif_cluster_assignments": MOTIF_CLUSTER_ASSIGNMENTS_FILE,
    "protein_similarity_long": PROTEIN_SIMILARITY_LONG_FILE,
    "motif_cluster_protein_counts": MOTIF_CLUSTER_PROTEIN_COUNTS_FILE,
}

WINDOW_FEATURES_REQUIRED_COLUMNS = [
    "protein",
    "chain",
    "segment_label",
    "window_label",
    "window_center_seq_index",
]
MOTIF_ASSIGNMENTS_REQUIRED_COLUMNS = ["motif_cluster_id", "protein", "chain", "window_label"]
SIMILARITY_LONG_REQUIRED_COLUMNS = [
    "protein_a",
    "protein_b",
    "symmetric_distance",
    "similarity_score",
]
CLUSTER_COUNTS_REQUIRED_COLUMNS = [
    "motif_cluster_id",
    "protein",
    "n_windows",
    "fraction_within_cluster",
]

FIG_DPI = 300


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the comparison plotting module."""

    parser = argparse.ArgumentParser(
        description=(
            "Create publication-quality plots from precomputed protein and motif "
            "comparison tables."
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
        help="Directory where plot files will be written.",
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
    """Load all required comparison tables from disk."""

    data: dict[str, pd.DataFrame] = {}
    for key, filename in REQUIRED_INPUT_FILES.items():
        file_path = input_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Required input file not found: {file_path}")
        if key == "protein_similarity_matrix":
            data[key] = pd.read_csv(file_path, index_col=0)
        else:
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
    sim_matrix_df = data["protein_similarity_matrix"]
    sim_long_df = data["protein_similarity_long"]
    cluster_counts_df = data["motif_cluster_protein_counts"]

    missing_window_columns = [
        column for column in WINDOW_FEATURES_REQUIRED_COLUMNS if column not in window_df.columns
    ]
    if missing_window_columns:
        raise ValueError(
            f"{WINDOW_FEATURES_FILE}: missing required columns: "
            f"{', '.join(missing_window_columns)}"
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
            f"{', '.join(missing_motif_columns)}"
        )

    missing_long_columns = [
        column for column in SIMILARITY_LONG_REQUIRED_COLUMNS if column not in sim_long_df.columns
    ]
    if missing_long_columns:
        raise ValueError(
            f"{PROTEIN_SIMILARITY_LONG_FILE}: missing required columns: "
            f"{', '.join(missing_long_columns)}"
        )

    missing_cluster_columns = [
        column for column in CLUSTER_COUNTS_REQUIRED_COLUMNS if column not in cluster_counts_df.columns
    ]
    if missing_cluster_columns:
        raise ValueError(
            f"{MOTIF_CLUSTER_PROTEIN_COUNTS_FILE}: missing required columns: "
            f"{', '.join(missing_cluster_columns)}"
        )

    if window_df.empty:
        raise ValueError(f"{WINDOW_FEATURES_FILE}: input table is empty")
    if motif_df.empty:
        raise ValueError(f"{MOTIF_CLUSTER_ASSIGNMENTS_FILE}: input table is empty")
    if sim_matrix_df.empty:
        raise ValueError(f"{PROTEIN_SIMILARITY_MATRIX_FILE}: input table is empty")
    if sim_long_df.empty:
        raise ValueError(f"{PROTEIN_SIMILARITY_LONG_FILE}: input table is empty")
    if cluster_counts_df.empty:
        raise ValueError(f"{MOTIF_CLUSTER_PROTEIN_COUNTS_FILE}: input table is empty")

    protein_order = get_protein_order()
    matrix_rows = sim_matrix_df.index.astype(str).tolist()
    matrix_cols = sim_matrix_df.columns.astype(str).tolist()
    missing_matrix_rows = [protein for protein in protein_order if protein not in matrix_rows]
    missing_matrix_cols = [protein for protein in protein_order if protein not in matrix_cols]
    if missing_matrix_rows or missing_matrix_cols:
        details: list[str] = []
        if missing_matrix_rows:
            details.append(f"missing row proteins: {', '.join(missing_matrix_rows)}")
        if missing_matrix_cols:
            details.append(f"missing column proteins: {', '.join(missing_matrix_cols)}")
        raise ValueError(f"{PROTEIN_SIMILARITY_MATRIX_FILE}: {'; '.join(details)}")


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
        "cross-class": "#7B7D7D",
    }


def get_protein_order() -> list[str]:
    """Return the fixed plotting order for proteins."""

    return ["2HHB", "1GZM", "1TEN", "1FNA", "2IGF", "1PKK", "1UBQ", "1LYZ", "2PTN"]


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
    joint_sum = joint.sum()
    if joint_sum <= 0.0 or not np.isfinite(joint_sum):
        raise ValueError("Failed to compute a valid joint probability matrix for t-SNE")
    joint /= joint_sum
    joint = np.maximum(joint, 1e-12)
    joint /= joint.sum()
    return joint


def run_tsne(
    X: np.ndarray,
    perplexity: float,
    learning_rate: float,
    n_iter: int,
    random_seed: int,
    verbose: bool = False,
) -> np.ndarray:
    """Run exact t-SNE from scratch using NumPy only."""

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
    P /= P.sum()

    rng = np.random.default_rng(random_seed)
    Y = rng.normal(loc=0.0, scale=1e-4, size=(n_samples, 2))
    Y_incs = np.zeros_like(Y)
    gains = np.ones_like(Y)

    for iteration in range(n_iter):
        sum_Y = np.sum(Y * Y, axis=1)
        num = 1.0 / (1.0 + (-2.0 * (Y @ Y.T) + sum_Y[:, None] + sum_Y[None, :]))
        np.fill_diagonal(num, 0.0)
        Q = num / np.maximum(num.sum(), 1e-12)
        Q = np.maximum(Q, 1e-12)

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
            P /= P.sum()

        if verbose and ((iteration + 1) % 100 == 0 or iteration == 0 or iteration + 1 == n_iter):
            kl = float(np.sum(P * np.log(np.maximum(P, 1e-12) / Q)))
            print(f"t-SNE iteration {iteration + 1}/{n_iter}: KL={kl:.6f}")

    return Y


def get_motif_cluster_color_map(motif_cluster_ids: list[str]) -> dict[str, tuple[float, float, float, float]]:
    """Return a stable motif cluster palette."""

    cmap = plt.get_cmap("tab10")
    colors = [cmap(index % cmap.N) for index in range(len(motif_cluster_ids))]
    return {cluster_id: colors[index] for index, cluster_id in enumerate(motif_cluster_ids)}


def save_figure(fig: plt.Figure, png_path: Path, pdf_path: Path) -> tuple[Path, Path]:
    """Save a figure to both PNG and PDF."""

    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    return png_path, pdf_path


def style_scatter_axes(ax: plt.Axes) -> None:
    """Apply consistent scatter-axis styling."""

    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C8CED6")
    ax.spines["bottom"].set_color("#C8CED6")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(color="#89929B", labelcolor="#333333")


def plot_tsne_by_protein(tsne_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Plot the shared t-SNE embedding colored by protein."""

    protein_colors = get_protein_color_map()
    protein_order = get_protein_order()

    fig, ax = plt.subplots(figsize=(10, 8))
    for protein in protein_order:
        subset = tsne_df.loc[tsne_df["protein"] == protein]
        if subset.empty:
            continue
        ax.scatter(
            subset["tsne_1"],
            subset["tsne_2"],
            s=18,
            alpha=0.7,
            c=protein_colors[protein],
            edgecolors="white",
            linewidths=0.3,
            label=protein,
        )

    fig.suptitle("Theta-pp FFT Window Embedding (t-SNE)", fontsize=17, fontweight="bold", y=0.97)
    ax.set_title("colored by protein", fontsize=11, color="#56606B", pad=10)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    style_scatter_axes(ax)

    class_sections = [
        ("Helix", ["2HHB", "1GZM"]),
        ("Sheet", ["1TEN", "1FNA", "2IGF"]),
        ("Mixed", ["1PKK", "1UBQ", "1LYZ", "2PTN"]),
    ]
    handles: list[Any] = []
    labels: list[str] = []
    for section_label, proteins in class_sections:
        handles.append(Line2D([], [], linestyle="none"))
        labels.append(section_label)
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
    for text, handle in zip(legend.get_texts(), handles):
        if isinstance(handle, Line2D) and handle.get_marker() == "None":
            text.set_fontweight("bold")
            text.set_color("#2A2F35")

    fig.subplots_adjust(top=0.88, right=0.79)
    return save_figure(
        fig,
        output_dir / "tsne_by_protein.png",
        output_dir / "tsne_by_protein.pdf",
    )


def plot_tsne_by_motif_cluster(tsne_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Plot the shared t-SNE embedding colored by motif cluster."""

    motif_cluster_ids = sorted(tsne_df["motif_cluster_id"].astype(str).unique().tolist())
    cluster_colors = get_motif_cluster_color_map(motif_cluster_ids)

    fig, ax = plt.subplots(figsize=(10, 8))
    for cluster_id in motif_cluster_ids:
        subset = tsne_df.loc[tsne_df["motif_cluster_id"].astype(str) == cluster_id]
        if subset.empty:
            continue
        ax.scatter(
            subset["tsne_1"],
            subset["tsne_2"],
            s=18,
            alpha=0.72,
            c=[cluster_colors[cluster_id]],
            edgecolors="white",
            linewidths=0.3,
            label=cluster_id,
        )

    fig.suptitle("Theta-pp FFT Window Embedding (t-SNE)", fontsize=17, fontweight="bold", y=0.97)
    ax.set_title("colored by motif cluster", fontsize=11, color="#56606B", pad=10)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    style_scatter_axes(ax)

    n_clusters = len(motif_cluster_ids)
    ncol = 1 if n_clusters <= 10 else 2
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        title="Motif cluster",
        title_fontsize=10,
        ncol=ncol,
        scatterpoints=1,
        markerscale=1.2,
    )
    fig.subplots_adjust(top=0.88, right=0.78 if n_clusters <= 10 else 0.72)
    return save_figure(
        fig,
        output_dir / "tsne_by_motif_cluster.png",
        output_dir / "tsne_by_motif_cluster.pdf",
    )


def add_heatmap_class_bands(ax: plt.Axes, protein_order: list[str]) -> None:
    """Add tasteful structural-class accent bands to the heatmap axes."""

    accents = get_class_accent_colors()
    for index, protein in enumerate(protein_order):
        color = accents[get_protein_class(protein)]
        ax.add_patch(
            mpatches.Rectangle(
                (-0.5, index - 0.5),
                0.14,
                1.0,
                facecolor=color,
                edgecolor="none",
                clip_on=False,
                alpha=0.95,
                zorder=4,
            )
        )
        ax.add_patch(
            mpatches.Rectangle(
                (index - 0.5, -0.5),
                1.0,
                0.14,
                facecolor=color,
                edgecolor="none",
                clip_on=False,
                alpha=0.95,
                zorder=4,
            )
        )


def plot_protein_similarity_heatmap(sim_matrix_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Plot the ordered protein similarity heatmap."""

    protein_order = get_protein_order()
    ordered = sim_matrix_df.copy()
    ordered.index = ordered.index.astype(str)
    ordered.columns = ordered.columns.astype(str)
    ordered = ordered.loc[protein_order, protein_order].apply(pd.to_numeric, errors="coerce")
    if ordered.isna().any().any():
        raise ValueError(f"{PROTEIN_SIMILARITY_MATRIX_FILE}: matrix contains non-numeric or missing values")

    values = ordered.to_numpy(dtype=float)
    cmap = plt.get_cmap("RdYlBu_r")
    norm = mcolors.Normalize(vmin=0.20, vmax=1.0)

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="equal")
    ax.set_title("Protein-to-Protein Spectral Similarity", loc="left", pad=14)
    ax.set_xticks(np.arange(len(protein_order)))
    ax.set_yticks(np.arange(len(protein_order)))
    ax.set_xticklabels(protein_order, rotation=45, ha="right")
    ax.set_yticklabels(protein_order)

    ax.set_xticks(np.arange(-0.5, len(protein_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(protein_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C8CED6")
    ax.spines["bottom"].set_color("#C8CED6")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            rgba = cmap(norm(values[i, j]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "black" if luminance > 0.57 else "white"
            ax.text(
                j,
                i,
                f"{values[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    add_heatmap_class_bands(ax, protein_order)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Similarity score")

    fig.tight_layout()
    return save_figure(
        fig,
        output_dir / "protein_similarity_heatmap.png",
        output_dir / "protein_similarity_heatmap.pdf",
    )


def classify_pair_type(protein_a: str, protein_b: str) -> str:
    """Classify a protein pair by structural class pairing."""

    class_a = get_protein_class(protein_a)
    class_b = get_protein_class(protein_b)
    if class_a == "helix" and class_b == "helix":
        return "helix-helix"
    if class_a == "sheet" and class_b == "sheet":
        return "sheet-sheet"
    if class_a == "mixed" and class_b == "mixed":
        return "mixed-mixed"
    return "cross-class"


def plot_distance_distributions(sim_long_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Plot spectral distances by structural class pairing for small sample sizes."""

    working_df = sim_long_df.copy()
    working_df["pair_type"] = [
        classify_pair_type(str(protein_a), str(protein_b))
        for protein_a, protein_b in zip(working_df["protein_a"], working_df["protein_b"])
    ]
    working_df["symmetric_distance"] = pd.to_numeric(
        working_df["symmetric_distance"], errors="coerce"
    )
    working_df = working_df.dropna(subset=["symmetric_distance"])

    pair_order = ["helix-helix", "sheet-sheet", "mixed-mixed", "cross-class"]
    pair_colors = {
        "helix-helix": get_class_accent_colors()["helix"],
        "sheet-sheet": get_class_accent_colors()["sheet"],
        "mixed-mixed": get_class_accent_colors()["mixed"],
        "cross-class": get_class_accent_colors()["cross-class"],
    }

    if working_df.empty:
        raise ValueError(f"{PROTEIN_SIMILARITY_LONG_FILE}: no valid symmetric_distance values were found")

    fig, ax = plt.subplots(figsize=(10, 6))
    rng = np.random.default_rng(42)
    x_positions = np.arange(len(pair_order), dtype=float)
    x_tick_labels: list[str] = []

    for index, pair_type in enumerate(pair_order):
        subset = working_df.loc[working_df["pair_type"] == pair_type, "symmetric_distance"]
        n_points = int(subset.shape[0])
        x_tick_labels.append(f"{pair_type}\n(n={n_points})")
        if subset.empty:
            continue

        values = subset.to_numpy(dtype=float)
        color = pair_colors[pair_type]
        jitter = rng.uniform(-0.12, 0.12, size=values.shape[0])
        x_values = np.full(values.shape[0], x_positions[index]) + jitter
        ax.scatter(
            x_values,
            values,
            s=52,
            alpha=0.78,
            c=color,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        mean_value = float(np.mean(values))
        ax.hlines(
            mean_value,
            x_positions[index] - 0.18,
            x_positions[index] + 0.18,
            color=color,
            linewidth=2.2,
            zorder=4,
        )
        ax.scatter(
            [x_positions[index]],
            [mean_value],
            s=120,
            c=color,
            edgecolors="#2F3640",
            linewidths=0.9,
            zorder=5,
        )

    ax.set_title("Spectral Distances by Structural Class Pairing", loc="left", pad=14)
    ax.set_xlabel("Structural class pairing")
    ax.set_ylabel("Symmetric distance (z-space)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_tick_labels)
    ax.grid(axis="y", color="#DCE2E8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C8CED6")
    ax.spines["bottom"].set_color("#C8CED6")
    ax.set_xlim(-0.5, len(pair_order) - 0.5)

    fig.tight_layout()
    return save_figure(
        fig,
        output_dir / "distance_distributions.png",
        output_dir / "distance_distributions.pdf",
    )


def plot_motif_cluster_composition(
    cluster_counts_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Plot motif cluster protein composition as stacked horizontal bars."""

    protein_order = get_protein_order()
    protein_colors = get_protein_color_map()
    working_df = cluster_counts_df.copy()
    working_df["motif_cluster_id"] = working_df["motif_cluster_id"].astype(str)
    working_df["protein"] = working_df["protein"].astype(str)
    working_df["n_windows"] = pd.to_numeric(working_df["n_windows"], errors="coerce")
    working_df["fraction_within_cluster"] = pd.to_numeric(
        working_df["fraction_within_cluster"], errors="coerce"
    )
    working_df = working_df.dropna(subset=["n_windows", "fraction_within_cluster"])

    cluster_order = sorted(working_df["motif_cluster_id"].unique().tolist())
    pivot = (
        working_df.pivot_table(
            index="motif_cluster_id",
            columns="protein",
            values="fraction_within_cluster",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=cluster_order, columns=protein_order, fill_value=0.0)
    )
    totals = (
        working_df.groupby("motif_cluster_id", sort=False)["n_windows"]
        .sum()
        .reindex(cluster_order)
        .astype(int)
    )

    fig_height = max(6.0, 0.6 * len(cluster_order) + 1.8)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y_positions = np.arange(len(cluster_order))
    left = np.zeros(len(cluster_order), dtype=float)

    for protein in protein_order:
        widths = pivot[protein].to_numpy(dtype=float)
        if np.allclose(widths, 0.0):
            continue
        ax.barh(
            y_positions,
            widths,
            left=left,
            height=0.72,
            color=protein_colors[protein],
            edgecolor="white",
            linewidth=0.8,
            label=protein,
        )
        left += widths

    y_tick_labels = [f"{cluster_id}  (n={totals.loc[cluster_id]})" for cluster_id in cluster_order]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_tick_labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(1.02, float(np.nanmax(left)) + 0.08))
    ax.set_xlabel("Fraction of windows within cluster")
    ax.set_ylabel("Motif cluster")
    ax.set_title("Motif Cluster Protein Composition", loc="left", pad=14)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C8CED6")
    ax.spines["bottom"].set_color("#C8CED6")

    for y_pos, total_width, cluster_id in zip(y_positions, left, cluster_order):
        ax.text(
            min(total_width + 0.015, ax.get_xlim()[1] - 0.01),
            y_pos,
            f"n={totals.loc[cluster_id]}",
            va="center",
            ha="left",
            fontsize=9,
            color="#3D4650",
        )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        title="Protein",
        title_fontsize=10,
    )
    fig.subplots_adjust(right=0.80)
    return save_figure(
        fig,
        output_dir / "motif_cluster_composition.png",
        output_dir / "motif_cluster_composition.pdf",
    )


def main() -> None:
    """Run the standalone comparison plotting workflow."""

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
        print(f"Using {len(z_columns)} standardized feature columns for t-SNE")

    missing_feature_mask = window_df[z_columns].isna().any(axis=1)
    n_dropped = int(missing_feature_mask.sum())
    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows with missing z__ features before t-SNE")
    tsne_input_df = window_df.loc[~missing_feature_mask].copy()

    if tsne_input_df.empty:
        raise ValueError("No rows remained for t-SNE after dropping missing feature values")

    X = tsne_input_df[z_columns].to_numpy(dtype=float)
    print(f"Windows used for t-SNE: {X.shape[0]}")

    tsne_coordinates = run_tsne(
        X=X,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        random_seed=args.random_seed,
        verbose=args.verbose,
    )

    tsne_df = tsne_input_df.loc[:, ["window_label", "protein", "chain", "segment_label"]].copy()
    tsne_df["window_label"] = tsne_df["window_label"].astype(str)
    tsne_df["protein"] = tsne_df["protein"].astype(str)
    tsne_df["chain"] = tsne_df["chain"].astype(str)
    tsne_df["segment_label"] = tsne_df["segment_label"].astype(str)
    tsne_df["tsne_1"] = tsne_coordinates[:, 0]
    tsne_df["tsne_2"] = tsne_coordinates[:, 1]

    motif_merge = motif_df.loc[:, ["window_label", "motif_cluster_id"]].copy()
    motif_merge["window_label"] = motif_merge["window_label"].astype(str)
    duplicate_windows = motif_merge["window_label"].duplicated()
    if duplicate_windows.any():
        duplicates = motif_merge.loc[duplicate_windows, "window_label"].astype(str).head(5).tolist()
        raise ValueError(
            f"{MOTIF_CLUSTER_ASSIGNMENTS_FILE}: expected one row per plotted window_label; duplicate examples: "
            f"{', '.join(duplicates)}"
        )

    merged_tsne_df = tsne_df.merge(
        motif_merge,
        on="window_label",
        how="left",
        validate="one_to_one",
    )
    if merged_tsne_df["motif_cluster_id"].isna().any():
        missing_count = int(merged_tsne_df["motif_cluster_id"].isna().sum())
        raise ValueError(
            f"{MOTIF_CLUSTER_ASSIGNMENTS_FILE}: motif_cluster_id missing for {missing_count} plotted windows after merge"
        )

    if args.verbose:
        print("Plotting t-SNE embedding colored by protein")
    plot_tsne_by_protein(merged_tsne_df, output_dir)

    if args.verbose:
        print("Plotting t-SNE embedding colored by motif cluster")
    plot_tsne_by_motif_cluster(merged_tsne_df, output_dir)

    if args.verbose:
        print("Plotting protein similarity heatmap")
    plot_protein_similarity_heatmap(data["protein_similarity_matrix"], output_dir)

    if args.verbose:
        print("Plotting structural-class distance distributions")
    plot_distance_distributions(data["protein_similarity_long"], output_dir)

    if args.verbose:
        print("Plotting motif cluster protein composition")
    plot_motif_cluster_composition(data["motif_cluster_protein_counts"], output_dir)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(
            "plot_comparison.py was created successfully, but it could not be run because the "
            f"required comparison outputs were missing: {exc}"
        )
        raise
