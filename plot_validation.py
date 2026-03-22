"""Visualize precomputed Module 4 validation outputs for Theta_FFT.

This script is intentionally separate from analysis by design. It reads
already-generated validation tables, creates polished cluster-level,
class-level, protein-level, and window-level validation figures, and can
reuse or deterministically recreate the shared t-SNE coordinates for the
DSSP overlay. It does not recompute FFT, clustering, or DSSP labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLUSTER_LABEL_MAPPING_FILE = "cluster_label_mapping.csv"
CLUSTER_PURITY_FILE = "cluster_purity.csv"
CONFUSION_MATRIX_FILE = "confusion_matrix.csv"
PRECISION_RECALL_FILE = "precision_recall_by_class.csv"
WINDOW_VALIDATION_FILE = "window_validation.csv"
SKIPPED_WINDOWS_FILE = "skipped_windows.csv"
WINDOW_FEATURES_FILE = "window_features_standardized.csv"
WINDOW_TSNE_COORDINATES_FILE = "window_tsne_coordinates.csv"

VALID_DSSP_LABELS = ["H", "E", "L"]
FIG_DPI = 300

REQUIRED_VALIDATION_FILES: dict[str, str] = {
    "cluster_label_mapping": CLUSTER_LABEL_MAPPING_FILE,
    "cluster_purity": CLUSTER_PURITY_FILE,
    "confusion_matrix": CONFUSION_MATRIX_FILE,
    "precision_recall": PRECISION_RECALL_FILE,
    "window_validation": WINDOW_VALIDATION_FILE,
    "skipped_windows": SKIPPED_WINDOWS_FILE,
}

REQUIRED_COMPARISON_FILES: dict[str, str] = {
    "window_features": WINDOW_FEATURES_FILE,
}

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "cluster_label_mapping": [
        "motif_cluster_id",
        "predicted_structural_class",
        "n_total_windows_in_cluster",
        "n_validated_windows_in_cluster",
        "count_H",
        "count_E",
        "count_L",
        "majority_fraction",
        "mapping_status",
    ],
    "cluster_purity": [
        "motif_cluster_id",
        "dssp_label",
        "n_windows_with_label",
        "n_validated_windows_in_cluster",
        "n_total_windows_in_cluster",
        "fraction_of_validated_windows",
        "fraction_of_total_windows",
        "cluster_assigned_class",
        "cluster_majority_fraction",
    ],
    "confusion_matrix": ["motif_cluster_id", "H", "E", "L"],
    "precision_recall": [
        "structural_class",
        "true_positive",
        "false_positive",
        "false_negative",
        "precision",
        "recall",
        "f1",
        "support_ground_truth",
        "support_predicted",
        "n_clusters_assigned_to_class",
        "n_scored_windows",
    ],
    "window_validation": [
        "window_label",
        "protein",
        "chain",
        "motif_cluster_id",
        "window_start_seq_index",
        "window_end_seq_index",
        "window_center_seq_index",
        "validation_status",
        "dssp_majority_label",
        "dssp_majority_fraction",
        "validation_eligible_for_metrics",
        "dssp_count_H",
        "dssp_count_E",
        "dssp_count_L",
    ],
    "skipped_windows": ["window_label", "protein", "chain", "motif_cluster_id", "reason"],
    "window_features": ["window_label", "protein", "chain", "segment_label"],
    "window_tsne_coordinates": ["window_label", "tsne_1", "tsne_2"],
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for validation plotting."""

    parser = argparse.ArgumentParser(
        description=(
            "Create publication-quality validation figures from precomputed "
            "Theta_FFT Module 4 outputs."
        )
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("Theta_FFT/output/validation"),
        help="Directory containing precomputed validation CSV outputs.",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=Path("Theta_FFT/output/comparison"),
        help="Directory containing comparison outputs and standardized features.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Theta_FFT/output/validation_plots"),
        help="Directory where plot files will be written.",
    )
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=200.0,
        help="t-SNE learning rate.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=1000,
        help="Number of t-SNE optimization iterations.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for deterministic t-SNE and jitter.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print extra plotting progress.")
    return parser.parse_args()


def ensure_output_dir(output_dir: Path) -> None:
    """Create the validation plot output directory if needed."""

    output_dir.mkdir(parents=True, exist_ok=True)


def load_inputs(validation_dir: Path, comparison_dir: Path) -> dict[str, pd.DataFrame | Path]:
    """Load all required validation and comparison inputs from disk."""

    data: dict[str, pd.DataFrame | Path] = {}

    for key, filename in REQUIRED_VALIDATION_FILES.items():
        file_path = validation_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Required input file not found: {file_path}")
        data[key] = pd.read_csv(file_path)
        print(f"Loaded input: {file_path}")

    for key, filename in REQUIRED_COMPARISON_FILES.items():
        file_path = comparison_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Required input file not found: {file_path}")
        data[key] = pd.read_csv(file_path)
        print(f"Loaded input: {file_path}")

    tsne_coordinate_path = comparison_dir / WINDOW_TSNE_COORDINATES_FILE
    data["window_tsne_coordinates_path"] = tsne_coordinate_path
    if tsne_coordinate_path.exists():
        print(f"Found saved t-SNE coordinate file: {tsne_coordinate_path}")
    else:
        print(
            "Saved t-SNE coordinate file not found; coordinates will be recreated "
            f"deterministically from {comparison_dir / WINDOW_FEATURES_FILE}"
        )

    return data


def validate_inputs(data: dict[str, pd.DataFrame | Path]) -> None:
    """Validate the presence and schema of all required plotting inputs."""

    for key in list(REQUIRED_VALIDATION_FILES) + list(REQUIRED_COMPARISON_FILES):
        if key not in data:
            raise ValueError(f"Missing loaded input table: {key}")

    for key, required_columns in REQUIRED_COLUMNS.items():
        if key == "window_tsne_coordinates":
            continue
        frame = data.get(key)
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"Expected DataFrame for input '{key}'")
        missing_columns = [column for column in required_columns if column not in frame.columns]
        if missing_columns:
            source_name = REQUIRED_VALIDATION_FILES.get(key) or REQUIRED_COMPARISON_FILES.get(key) or key
            raise ValueError(
                f"{source_name}: missing required columns: {', '.join(missing_columns)}"
            )

    window_features_df = data["window_features"]
    if not isinstance(window_features_df, pd.DataFrame):
        raise ValueError("window_features input was not loaded as a DataFrame")
    z_columns = [column for column in window_features_df.columns if column.startswith("z__")]
    if not z_columns:
        raise ValueError(
            f"{WINDOW_FEATURES_FILE}: no standardized feature columns prefixed with 'z__' were found"
        )

    for key, value in data.items():
        if key.endswith("_path"):
            continue
        if isinstance(value, pd.DataFrame) and key != "skipped_windows" and value.empty:
            source_name = REQUIRED_VALIDATION_FILES.get(key) or REQUIRED_COMPARISON_FILES.get(key) or key
            raise ValueError(f"{source_name}: input table is empty")

    coord_path = data.get("window_tsne_coordinates_path")
    if isinstance(coord_path, Path) and coord_path.exists():
        coord_df = pd.read_csv(coord_path)
        missing_columns = [
            column
            for column in REQUIRED_COLUMNS["window_tsne_coordinates"]
            if column not in coord_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"{WINDOW_TSNE_COORDINATES_FILE}: missing required columns: "
                f"{', '.join(missing_columns)}"
            )


def get_protein_order() -> list[str]:
    """Return the fixed plotting order for proteins."""

    return ["2HHB", "1GZM", "1TEN", "1FNA", "2IGF", "1PKK", "1UBQ", "1LYZ", "2PTN"]


def get_protein_class(protein: str) -> str:
    """Return the canonical structural grouping for a protein."""

    helix = {"2HHB", "1GZM"}
    sheet = {"1TEN", "1FNA", "2IGF"}
    mixed = {"1PKK", "1UBQ", "1LYZ", "2PTN"}

    if protein in helix:
        return "helix"
    if protein in sheet:
        return "sheet"
    if protein in mixed:
        return "mixed"
    raise ValueError(f"Unknown protein for class assignment: {protein}")


def get_protein_class_colors() -> dict[str, str]:
    """Return structural-group accent colors for proteins."""

    return {
        "helix": "#D35400",
        "sheet": "#2471A3",
        "mixed": "#6E7B34",
    }


def get_dssp_class_colors() -> dict[str, str]:
    """Return the shared DSSP color palette used across figures."""

    return {
        "H": "#D65F5F",
        "E": "#3B6FB6",
        "L": "#7A8F55",
        "ambiguous": "#C7CCD1",
        "neutral_dark": "#5F6B73",
    }


def get_metric_colors() -> dict[str, str]:
    """Return the fixed color palette for precision, recall, and F1."""

    return {
        "precision": "#4C78A8",
        "recall": "#59A14F",
        "f1": "#E3A008",
    }


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
    """Find conditional probabilities whose entropy matches the target perplexity."""

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


def compute_joint_probabilities(
    X: np.ndarray, perplexity: float, verbose: bool = False
) -> np.ndarray:
    """Compute the symmetric joint probability matrix for exact t-SNE."""

    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("t-SNE requires at least two samples")
    if perplexity >= n_samples:
        raise ValueError(
            f"perplexity must be smaller than the number of samples; got {perplexity} "
            f"for {n_samples} samples"
        )

    distances = compute_pairwise_squared_distances(X)
    conditional = np.zeros((n_samples, n_samples), dtype=float)

    for i in range(n_samples):
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        row_probabilities, sigma = binary_search_sigma(distances[i, mask], perplexity)
        conditional[i, mask] = row_probabilities
        if verbose and ((i + 1) % 100 == 0 or i + 1 == n_samples):
            print(
                f"Computed conditional probabilities for {i + 1}/{n_samples} points "
                f"(sigma={sigma:.4f})"
            )

    joint = conditional + conditional.T
    joint_sum = joint.sum()
    if joint_sum <= 0.0 or not np.isfinite(joint_sum):
        raise ValueError("Failed to compute a valid t-SNE probability matrix")
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
    """Run exact NumPy-only t-SNE using the comparison module's deterministic logic."""

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

        if verbose and (
            iteration == 0 or (iteration + 1) % 100 == 0 or iteration + 1 == n_iter
        ):
            kl_divergence = float(np.sum(P * np.log(np.maximum(P, 1e-12) / Q)))
            print(f"t-SNE iteration {iteration + 1}/{n_iter}: KL={kl_divergence:.6f}")

    return Y


def load_or_create_tsne_coordinates(
    window_features_df: pd.DataFrame,
    comparison_dir: Path,
    perplexity: float,
    learning_rate: float,
    n_iter: int,
    random_seed: int,
    verbose: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Load saved t-SNE coordinates or deterministically recreate them from z__ features."""

    coord_path = comparison_dir / WINDOW_TSNE_COORDINATES_FILE
    if coord_path.exists():
        coord_df = pd.read_csv(coord_path)
        missing_columns = [
            column
            for column in REQUIRED_COLUMNS["window_tsne_coordinates"]
            if column not in coord_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"{coord_path.name}: missing required columns: {', '.join(missing_columns)}"
            )
        if coord_df["window_label"].duplicated().any():
            duplicate_examples = (
                coord_df.loc[coord_df["window_label"].duplicated(), "window_label"]
                .astype(str)
                .head(5)
                .tolist()
            )
            raise ValueError(
                f"{coord_path.name}: duplicate window_label values found; examples: "
                f"{', '.join(duplicate_examples)}"
            )

        coord_df = coord_df.loc[:, ["window_label", "tsne_1", "tsne_2"]].copy()
        coord_df["window_label"] = coord_df["window_label"].astype(str)
        coord_df["tsne_1"] = pd.to_numeric(coord_df["tsne_1"], errors="coerce")
        coord_df["tsne_2"] = pd.to_numeric(coord_df["tsne_2"], errors="coerce")
        if coord_df[["tsne_1", "tsne_2"]].isna().any().any():
            raise ValueError(f"{coord_path.name}: non-numeric or missing t-SNE coordinates found")
        print(f"t-SNE coordinate source: saved coordinates ({coord_path})")
        return coord_df, "saved"

    z_columns = sorted(
        [column for column in window_features_df.columns if column.startswith("z__")]
    )
    missing_feature_mask = window_features_df[z_columns].isna().any(axis=1)
    n_dropped = int(missing_feature_mask.sum())
    if n_dropped > 0:
        print(f"Dropped {n_dropped} windows with missing z__ feature values before t-SNE")

    tsne_input_df = window_features_df.loc[~missing_feature_mask].copy()
    if tsne_input_df.empty:
        raise ValueError("No rows remained for t-SNE after dropping missing z__ feature values")
    if tsne_input_df["window_label"].duplicated().any():
        duplicate_examples = (
            tsne_input_df.loc[tsne_input_df["window_label"].duplicated(), "window_label"]
            .astype(str)
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"{WINDOW_FEATURES_FILE}: duplicate window_label values found; examples: "
            f"{', '.join(duplicate_examples)}"
        )

    X = tsne_input_df[z_columns].to_numpy(dtype=float)
    if verbose:
        print(f"Using {len(z_columns)} standardized features and {X.shape[0]} windows for t-SNE")

    coords = run_tsne(
        X=X,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_seed=random_seed,
        verbose=verbose,
    )

    coord_df = tsne_input_df.loc[:, ["window_label"]].copy()
    coord_df["window_label"] = coord_df["window_label"].astype(str)
    coord_df["tsne_1"] = coords[:, 0]
    coord_df["tsne_2"] = coords[:, 1]
    print(f"t-SNE coordinate source: regenerated deterministically from {WINDOW_FEATURES_FILE}")
    return coord_df, "regenerated"


def build_cluster_purity_matrix(cluster_purity_df: pd.DataFrame) -> pd.DataFrame:
    """Build the motif-cluster-by-DSSP purity matrix for plotting."""

    working_df = cluster_purity_df.copy()
    working_df["motif_cluster_id"] = working_df["motif_cluster_id"].astype(str)
    working_df["dssp_label"] = working_df["dssp_label"].astype(str)
    working_df["fraction_of_validated_windows"] = pd.to_numeric(
        working_df["fraction_of_validated_windows"], errors="coerce"
    )

    matrix = (
        working_df.pivot_table(
            index="motif_cluster_id",
            columns="dssp_label",
            values="fraction_of_validated_windows",
            aggfunc="first",
        )
        .reindex(index=sorted(working_df["motif_cluster_id"].unique().tolist()))
        .reindex(columns=VALID_DSSP_LABELS)
        .fillna(0.0)
    )
    matrix.index.name = "motif_cluster_id"
    return matrix


def build_per_protein_dssp_composition(window_validation_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate validation-eligible H/E/L fractions for each protein."""

    eligible_df = window_validation_df.copy()
    eligible_df["validation_eligible_for_metrics"] = _coerce_bool_series(
        eligible_df["validation_eligible_for_metrics"]
    )
    eligible_df = eligible_df.loc[
        (eligible_df["validation_eligible_for_metrics"] == True)
        & (eligible_df["dssp_majority_label"].astype(str).isin(VALID_DSSP_LABELS))
    ].copy()

    protein_order = get_protein_order()
    counts = (
        eligible_df.groupby(["protein", "dssp_majority_label"], dropna=False)
        .size()
        .rename("n_windows")
        .reset_index()
    )
    pivot = (
        counts.pivot_table(
            index="protein",
            columns="dssp_majority_label",
            values="n_windows",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=protein_order, columns=VALID_DSSP_LABELS, fill_value=0)
        .astype(float)
    )
    totals = pivot.sum(axis=1)
    fractions = pivot.div(totals.replace(0.0, np.nan), axis=0).fillna(0.0)
    result = fractions.reset_index()
    result["n_eligible_windows"] = (
        totals.reindex(protein_order).fillna(0.0).astype(int).to_numpy()
    )
    return result


def build_per_protein_accuracy(
    window_validation_df: pd.DataFrame,
    cluster_label_mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-protein agreement between cluster predictions and DSSP majority labels."""

    working_df = window_validation_df.copy()
    working_df["validation_eligible_for_metrics"] = _coerce_bool_series(
        working_df["validation_eligible_for_metrics"]
    )
    mapping_df = cluster_label_mapping_df.loc[
        :, ["motif_cluster_id", "predicted_structural_class"]
    ].copy()
    mapping_df["motif_cluster_id"] = mapping_df["motif_cluster_id"].astype(str)

    merged = working_df.merge(
        mapping_df,
        on="motif_cluster_id",
        how="left",
        validate="many_to_one",
    )
    scored = merged.loc[
        (merged["validation_eligible_for_metrics"] == True)
        & (merged["dssp_majority_label"].astype(str).isin(VALID_DSSP_LABELS))
        & (merged["predicted_structural_class"].astype(str).isin(VALID_DSSP_LABELS))
    ].copy()

    scored["is_correct"] = (
        scored["predicted_structural_class"].astype(str)
        == scored["dssp_majority_label"].astype(str)
    )

    protein_order = get_protein_order()
    summary = (
        scored.groupby("protein", dropna=False)
        .agg(
            n_scored_windows=("window_label", "size"),
            n_correct=("is_correct", "sum"),
        )
        .reindex(protein_order)
    )
    summary["n_scored_windows"] = summary["n_scored_windows"].fillna(0.0).astype(int)
    summary["n_correct"] = summary["n_correct"].fillna(0.0).astype(int)
    summary["accuracy"] = np.where(
        summary["n_scored_windows"] > 0,
        summary["n_correct"] / summary["n_scored_windows"],
        np.nan,
    )
    return summary.reset_index().rename(columns={"index": "protein"})


def build_skipped_window_summaries(
    skipped_windows_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build skipped-window count summaries for proteins and motif clusters."""

    working_df = skipped_windows_df.copy()
    if not working_df.empty:
        working_df["protein"] = working_df["protein"].astype(str)
        working_df["motif_cluster_id"] = working_df["motif_cluster_id"].astype(str)

    protein_order = get_protein_order()
    protein_counts = (
        working_df.groupby("protein", dropna=False)
        .size()
        .rename("skipped_count")
        .reindex(protein_order, fill_value=0)
        .reset_index()
    )

    if working_df.empty:
        cluster_counts = pd.DataFrame(columns=["motif_cluster_id", "skipped_count"])
    else:
        cluster_counts = (
            working_df.groupby("motif_cluster_id", dropna=False)
            .size()
            .rename("skipped_count")
            .sort_index()
            .reset_index()
        )

    return protein_counts, cluster_counts


def deterministic_jitter(n: int, scale: float = 0.08, seed: int = 42) -> np.ndarray:
    """Return deterministic jitter offsets for scatter overlays."""

    if n <= 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=n)


def _get_dataframe(data: dict[str, pd.DataFrame | Path], key: str) -> pd.DataFrame:
    """Return a loaded DataFrame and enforce the expected type."""

    value = data.get(key)
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"Expected DataFrame for key '{key}'")
    return value.copy()


def _resolve_cli_path(path_value: Path, script_dir: Path) -> Path:
    """Resolve CLI paths relative to cwd, the script directory, or its parent."""

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


def _configure_matplotlib_style() -> None:
    """Apply a clean, publication-oriented matplotlib style."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11.5,
            "axes.edgecolor": "#C5CDD5",
            "axes.linewidth": 0.9,
            "axes.labelcolor": "#27313A",
            "xtick.color": "#404A53",
            "ytick.color": "#404A53",
            "xtick.labelsize": 9.7,
            "ytick.labelsize": 9.7,
            "grid.color": "#E3E8EE",
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#D6DCE3",
            "legend.framealpha": 0.98,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    """Convert bool-like CSV values into a clean boolean Series."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1", "yes"])


def _style_axes(ax: plt.Axes, axis: str | None = None, show_grid: bool = True) -> None:
    """Apply consistent, low-clutter axis styling."""

    ax.set_axisbelow(True)
    if show_grid:
        if axis is None:
            ax.grid(True)
        else:
            ax.grid(axis=axis)
    else:
        ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C5CDD5")
    ax.spines["bottom"].set_color("#C5CDD5")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)


def _save_figure(fig: plt.Figure, png_path: Path, pdf_path: Path) -> tuple[Path, Path]:
    """Save a figure to both PNG and PDF with the configured output quality."""

    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    return png_path, pdf_path


def _add_protein_group_accents(ax: plt.Axes, protein_order: list[str]) -> None:
    """Add subtle structural-group accent lines above protein-based bar charts."""

    del protein_order
    accent_colors = get_protein_class_colors()
    groups = [
        ("Helix", 0, 1, accent_colors["helix"]),
        ("Sheet", 2, 4, accent_colors["sheet"]),
        ("Mixed", 5, 8, accent_colors["mixed"]),
    ]
    transform = ax.get_xaxis_transform()
    for label, start, end, color in groups:
        ax.plot(
            [start - 0.34, end + 0.34],
            [1.004, 1.004],
            color=color,
            linewidth=2.2,
            transform=transform,
            clip_on=False,
        )
        ax.text(
            (start + end) / 2.0,
            1.022,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
            fontweight="bold",
            transform=transform,
            clip_on=False,
        )


def _add_figure_header(
    fig: plt.Figure,
    title: str,
    subtitle: str | None = None,
    *,
    title_y: float = 0.975,
    subtitle_y: float = 0.922,
    left: float = 0.125,
) -> None:
    """Add a clean title/subtitle block with consistent spacing."""

    fig.suptitle(title, fontsize=16, fontweight="bold", x=left, ha="left", y=title_y)
    if subtitle:
        fig.text(
            left,
            subtitle_y,
            subtitle,
            ha="left",
            va="center",
            fontsize=10.4,
            color="#5F6B73",
        )


def _style_scatter_axes(ax: plt.Axes) -> None:
    """Match the scatter-axis styling used by the comparison t-SNE figures."""

    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C8CED6")
    ax.spines["bottom"].set_color("#C8CED6")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(color="#89929B", labelcolor="#333333")


def plot_cluster_purity_heatmap(
    cluster_purity_df: pd.DataFrame,
    output_dir: Path,
    verbose: bool = False,
) -> tuple[Path, Path]:
    """Plot the cluster-purity headline heatmap."""

    if verbose:
        print("Plotting cluster purity heatmap")

    colors = get_dssp_class_colors()
    matrix = build_cluster_purity_matrix(cluster_purity_df)
    counts_matrix = (
        cluster_purity_df.pivot_table(
            index="motif_cluster_id",
            columns="dssp_label",
            values="n_windows_with_label",
            aggfunc="first",
        )
        .reindex(index=matrix.index, columns=VALID_DSSP_LABELS)
        .fillna(0)
        .astype(int)
    )
    row_annotations = (
        cluster_purity_df.loc[
            :, ["motif_cluster_id", "cluster_assigned_class", "cluster_majority_fraction"]
        ]
        .drop_duplicates(subset=["motif_cluster_id"])
        .set_index("motif_cluster_id")
        .reindex(matrix.index)
    )

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "purity_slate", ["#FBFCFE", "#DCE5F4", "#8EA6CA", "#334B6B"]
    )

    fig, (ax, ann_ax, cax) = plt.subplots(
        ncols=3,
        figsize=(8, 5.5),
        gridspec_kw={"width_ratios": [5.1, 1.75, 0.24]},
    )
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(VALID_DSSP_LABELS)))
    ax.set_xticklabels(VALID_DSSP_LABELS)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index.tolist())
    ax.set_xlabel("DSSP majority label")
    ax.set_ylabel("Motif cluster")
    ax.set_xticks(np.arange(-0.5, len(VALID_DSSP_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="#E5EAF0", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    _style_axes(ax, show_grid=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for row_index, cluster_id in enumerate(matrix.index):
        for col_index, dssp_label in enumerate(VALID_DSSP_LABELS):
            value = float(matrix.loc[cluster_id, dssp_label])
            count = int(counts_matrix.loc[cluster_id, dssp_label])
            rgba = cmap(value)
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "white" if luminance < 0.47 else "#24303C"
            ax.text(
                col_index,
                row_index - 0.10,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=text_color,
            )
            ax.text(
                col_index,
                row_index + 0.20,
                f"n={count}",
                ha="center",
                va="center",
                fontsize=7.3,
                color=text_color,
                alpha=0.95,
            )

    ann_ax.set_xlim(0.0, 1.0)
    ann_ax.set_ylim(len(matrix.index) - 0.5, -0.5)
    ann_ax.axis("off")
    ann_ax.text(
        0.02,
        -0.52,
        "Assigned class",
        fontsize=9,
        fontweight="bold",
        color=colors["neutral_dark"],
        ha="left",
        va="bottom",
    )
    for row_index, cluster_id in enumerate(matrix.index):
        assigned = str(row_annotations.loc[cluster_id, "cluster_assigned_class"])
        majority_fraction = row_annotations.loc[cluster_id, "cluster_majority_fraction"]
        label_color = colors.get(assigned, colors["ambiguous"])
        majority_text = "NA" if pd.isna(majority_fraction) else f"{float(majority_fraction):.3f}"
        ann_ax.text(
            0.02,
            row_index,
            assigned,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=label_color,
        )
        ann_ax.text(
            0.32,
            row_index,
            f"f={majority_text}",
            ha="left",
            va="center",
            fontsize=8.6,
            color=colors["neutral_dark"],
        )

    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Fraction of validated windows")
    colorbar.outline.set_edgecolor("#CAD1D8")

    _add_figure_header(
        fig,
        "Motif Cluster Purity by DSSP Majority Label",
        "fractions within cluster, using validation-eligible windows only",
        title_y=0.975,
        subtitle_y=0.922,
    )
    fig.subplots_adjust(top=0.82, wspace=0.16)
    return _save_figure(
        fig,
        output_dir / "cluster_purity_heatmap.png",
        output_dir / "cluster_purity_heatmap.pdf",
    )


def plot_cluster_dssp_composition(
    cluster_purity_df: pd.DataFrame,
    output_dir: Path,
    verbose: bool = False,
) -> tuple[Path, Path]:
    """Plot DSSP composition within each motif cluster as stacked horizontal bars."""

    if verbose:
        print("Plotting cluster DSSP composition")

    colors = get_dssp_class_colors()
    matrix = build_cluster_purity_matrix(cluster_purity_df)
    totals = (
        cluster_purity_df.loc[:, ["motif_cluster_id", "n_validated_windows_in_cluster"]]
        .drop_duplicates(subset=["motif_cluster_id"])
        .set_index("motif_cluster_id")
        .reindex(matrix.index)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    y_positions = np.arange(len(matrix.index))
    left = np.zeros(len(matrix.index), dtype=float)

    for label in VALID_DSSP_LABELS:
        widths = matrix[label].to_numpy(dtype=float)
        ax.barh(
            y_positions,
            widths,
            left=left,
            height=0.72,
            color=colors[label],
            edgecolor="white",
            linewidth=0.8,
            label=label,
        )
        left += widths

    y_labels = [
        f"{cluster_id}  (n={int(totals.loc[cluster_id, 'n_validated_windows_in_cluster'])})"
        for cluster_id in matrix.index
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Fraction of validated windows")
    ax.set_ylabel("Motif cluster")
    ax.set_title("DSSP Composition Within Each Motif Cluster", fontsize=16, fontweight="bold", pad=14)
    _style_axes(ax, axis="x")
    ax.legend(
        title="DSSP label",
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        borderpad=0.7,
        handlelength=1.2,
    )
    return _save_figure(
        fig,
        output_dir / "cluster_dssp_composition.png",
        output_dir / "cluster_dssp_composition.pdf",
    )


def plot_tsne_by_dssp_label(
    coord_df: pd.DataFrame,
    window_validation_df: pd.DataFrame,
    output_dir: Path,
    verbose: bool = False,
) -> tuple[Path, Path]:
    """Plot the shared t-SNE embedding recolored by DSSP majority label."""

    if verbose:
        print("Plotting t-SNE by DSSP majority label")

    colors = get_dssp_class_colors()

    coords = coord_df.copy()
    coords["window_label"] = coords["window_label"].astype(str)
    validation = window_validation_df.copy()
    validation["window_label"] = validation["window_label"].astype(str)
    validation["validation_eligible_for_metrics"] = _coerce_bool_series(
        validation["validation_eligible_for_metrics"]
    )

    if coords["window_label"].duplicated().any():
        raise ValueError("t-SNE coordinates contain duplicate window_label values")
    if validation["window_label"].duplicated().any():
        raise ValueError(f"{WINDOW_VALIDATION_FILE}: duplicate window_label values found")

    merged = coords.merge(
        validation.loc[:, ["window_label", "dssp_majority_label", "validation_eligible_for_metrics"]],
        on="window_label",
        how="left",
        validate="one_to_one",
    )
    if merged["dssp_majority_label"].isna().any():
        missing_count = int(merged["dssp_majority_label"].isna().sum())
        raise ValueError(
            f"{WINDOW_VALIDATION_FILE}: missing validation rows for {missing_count} t-SNE windows"
        )

    eligible_mask = (
        (merged["validation_eligible_for_metrics"] == True)
        & (merged["dssp_majority_label"].astype(str).isin(VALID_DSSP_LABELS))
    )
    ambiguous_df = merged.loc[~eligible_mask].copy()
    eligible_df = merged.loc[eligible_mask].copy()
    print(f"Windows plotted in t-SNE overlay: {len(merged)}")

    fig, ax = plt.subplots(figsize=(10, 8))

    if not ambiguous_df.empty:
        ax.scatter(
            ambiguous_df["tsne_1"],
            ambiguous_df["tsne_2"],
            s=18,
            alpha=0.28,
            c=colors["ambiguous"],
            edgecolors="white",
            linewidths=0.3,
            label="Ambiguous/Skipped",
            zorder=1,
        )

    for label in VALID_DSSP_LABELS:
        subset = eligible_df.loc[eligible_df["dssp_majority_label"].astype(str) == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["tsne_1"],
            subset["tsne_2"],
            s=18,
            alpha=0.78,
            c=colors[label],
            edgecolors="white",
            linewidths=0.3,
            label=label,
            zorder=2,
        )

    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    _style_scatter_axes(ax)

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=7.5,
            markerfacecolor=colors[label],
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=label,
        )
        for label in VALID_DSSP_LABELS
    ]
    legend_handles.append(
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=7.5,
            markerfacecolor=colors["ambiguous"],
            markeredgecolor="white",
            markeredgewidth=0.6,
            alpha=0.8,
            label="Ambiguous/Skipped",
        )
    )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        title="DSSP label",
        title_fontsize=10,
        handlelength=1.0,
        labelspacing=0.7,
    )

    fig.suptitle("Theta-pp FFT Window Embedding (t-SNE)", fontsize=17, fontweight="bold", y=0.97)
    ax.set_title(
        "colored by DSSP majority label",
        fontsize=11,
        color="#56606B",
        pad=10,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.88, right=0.79)
    return _save_figure(
        fig,
        output_dir / "tsne_by_dssp_label.png",
        output_dir / "tsne_by_dssp_label.pdf",
    )


def plot_class_metrics(
    precision_recall_df: pd.DataFrame,
    output_dir: Path,
    verbose: bool = False,
) -> tuple[Path, Path]:
    """Plot grouped precision, recall, and F1 scores by DSSP class."""

    if verbose:
        print("Plotting class-level validation metrics")

    metric_colors = get_metric_colors()
    metric_order = ["precision", "recall", "f1"]
    class_order = VALID_DSSP_LABELS
    metrics_df = precision_recall_df.copy()
    metrics_df["structural_class"] = metrics_df["structural_class"].astype(str)
    metrics_df = metrics_df.set_index("structural_class").reindex(class_order)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_positions = np.arange(len(class_order), dtype=float)
    width = 0.22

    for index, metric in enumerate(metric_order):
        x_values = x_positions + (index - 1) * width
        values = pd.to_numeric(metrics_df[metric], errors="coerce").to_numpy(dtype=float)
        for bar_x, value in zip(x_values, values):
            if np.isnan(value):
                ax.bar(
                    bar_x,
                    0.0,
                    width=width,
                    color="white",
                    edgecolor="#9EA7B1",
                    linewidth=1.0,
                    hatch="///",
                )
                ax.text(
                    bar_x,
                    0.025,
                    "NA",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color="#5F6B73",
                )
            else:
                ax.bar(
                    bar_x,
                    value,
                    width=width,
                    color=metric_colors[metric],
                    edgecolor="white",
                    linewidth=0.9,
                )
                ax.text(
                    bar_x,
                    min(value + 0.025, 1.02),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color="#36414A",
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_order)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Structural class")
    ax.set_ylabel("Score")
    ax.set_title("Validation Metrics by DSSP Class", fontsize=16, fontweight="bold", pad=14)
    _style_axes(ax, axis="y")
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=metric_colors["precision"], label="Precision"),
            mpatches.Patch(facecolor=metric_colors["recall"], label="Recall"),
            mpatches.Patch(facecolor=metric_colors["f1"], label="F1"),
        ],
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        borderpad=0.7,
    )
    fig.subplots_adjust(right=0.82)
    return _save_figure(
        fig,
        output_dir / "class_metrics.png",
        output_dir / "class_metrics.pdf",
    )


def plot_per_protein_dssp_composition(
    composition_df: pd.DataFrame,
    output_dir: Path,
    verbose: bool = False,
) -> tuple[Path, Path]:
    """Plot stacked H/E/L composition for each protein."""

    if verbose:
        print("Plotting per-protein DSSP composition")

    colors = get_dssp_class_colors()
    protein_order = get_protein_order()
    working_df = composition_df.set_index("protein").reindex(protein_order).reset_index()

    fig, ax = plt.subplots(figsize=(11, 5.8))
    x_positions = np.arange(len(protein_order), dtype=float)
    bottom = np.zeros(len(protein_order), dtype=float)

    for label in VALID_DSSP_LABELS:
        heights = pd.to_numeric(working_df[label], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ax.bar(
            x_positions,
            heights,
            width=0.74,
            bottom=bottom,
            color=colors[label],
            edgecolor="white",
            linewidth=0.8,
            label=label,
        )
        bottom += heights

    ax.set_xticks(x_positions)
    ax.set_xticklabels(protein_order)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Protein")
    ax.set_ylabel("Fraction of validation-eligible windows")
    _style_axes(ax, axis="y")
    _add_protein_group_accents(ax, protein_order)

    for x_pos, (_, row) in zip(x_positions, working_df.iterrows()):
        n_windows = int(row["n_eligible_windows"])
        if n_windows == 0:
            ax.text(
                x_pos,
                0.03,
                "n=0",
                ha="center",
                va="bottom",
                fontsize=8.3,
                color="#5F6B73",
            )
        else:
            ax.text(
                x_pos,
                min(float(row[VALID_DSSP_LABELS].sum()) + 0.02, 1.02),
                f"n={n_windows}",
                ha="center",
                va="bottom",
                fontsize=8.3,
                color="#3A434C",
            )

    ax.legend(
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        title="DSSP label",
    )
    _add_figure_header(
        fig,
        "Window-Level DSSP Composition by Protein",
        "fractions of validation-eligible windows colored by DSSP majority label",
        title_y=0.975,
        subtitle_y=0.922,
    )
    fig.subplots_adjust(top=0.82, right=0.82)
    return _save_figure(
        fig,
        output_dir / "per_protein_dssp_composition.png",
        output_dir / "per_protein_dssp_composition.pdf",
    )


def plot_per_protein_validation_accuracy(
    accuracy_df: pd.DataFrame,
    output_dir: Path,
    verbose: bool = False,
) -> tuple[Path, Path]:
    """Plot per-protein prediction agreement against DSSP majority labels."""

    if verbose:
        print("Plotting per-protein validation accuracy")

    protein_order = get_protein_order()
    accent_colors = get_protein_class_colors()
    working_df = accuracy_df.set_index("protein").reindex(protein_order).reset_index()

    bar_colors = [accent_colors[get_protein_class(protein)] for protein in protein_order]
    values = pd.to_numeric(working_df["accuracy"], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    x_positions = np.arange(len(protein_order), dtype=float)
    plot_values = np.nan_to_num(values, nan=0.0)
    bars = ax.bar(
        x_positions,
        plot_values,
        width=0.72,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.9,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(protein_order)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Protein")
    ax.set_ylabel("Accuracy")
    _style_axes(ax, axis="y")
    _add_protein_group_accents(ax, protein_order)

    for bar, value, (_, row) in zip(bars, values, working_df.iterrows()):
        x_pos = bar.get_x() + bar.get_width() / 2.0
        n_scored = int(row["n_scored_windows"])
        if np.isnan(value):
            ax.text(
                x_pos,
                0.03,
                "NA",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#5F6B73",
            )
            ax.text(
                x_pos,
                0.0,
                f"n={n_scored}",
                ha="center",
                va="top",
                fontsize=8.0,
                color="#5F6B73",
                transform=ax.get_xaxis_transform(),
            )
        else:
            ax.text(
                x_pos,
                min(float(value) + 0.025, 1.02),
                f"{float(value):.3f}",
                ha="center",
                va="bottom",
                fontsize=8.4,
                color="#36414A",
            )
            ax.text(
                x_pos,
                max(float(value) - 0.055, 0.04),
                f"n={n_scored}",
                ha="center",
                va="top",
                fontsize=7.8,
                color="white" if float(value) > 0.18 else "#4F5B64",
            )

    legend_handles = [
        mpatches.Patch(facecolor=accent_colors["helix"], label="Helix group"),
        mpatches.Patch(facecolor=accent_colors["sheet"], label="Sheet group"),
        mpatches.Patch(facecolor=accent_colors["mixed"], label="Mixed group"),
    ]
    ax.legend(
        handles=legend_handles,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    _add_figure_header(
        fig,
        "Per-Protein Agreement with DSSP Majority Labels",
        "accuracy over scored windows with resolved cluster-to-class predictions",
        title_y=0.975,
        subtitle_y=0.922,
    )
    fig.subplots_adjust(top=0.82, right=0.82)
    return _save_figure(
        fig,
        output_dir / "per_protein_validation_accuracy.png",
        output_dir / "per_protein_validation_accuracy.pdf",
    )


def plot_skipped_window_summary(
    protein_summary_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    output_dir: Path,
    verbose: bool = False,
) -> tuple[Path, Path]:
    """Plot skipped-window counts by protein and motif cluster."""

    if verbose:
        print("Plotting skipped-window summary")

    protein_order = get_protein_order()
    accent_colors = get_protein_class_colors()
    neutral_dark = get_dssp_class_colors()["neutral_dark"]
    protein_colors = [accent_colors[get_protein_class(protein)] for protein in protein_order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    ax_protein, ax_cluster = axes

    protein_summary = protein_summary_df.set_index("protein").reindex(protein_order).reset_index()
    protein_counts = pd.to_numeric(
        protein_summary["skipped_count"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    protein_bars = ax_protein.bar(
        np.arange(len(protein_order)),
        protein_counts,
        width=0.72,
        color=protein_colors,
        edgecolor="white",
        linewidth=0.8,
    )
    ax_protein.set_xticks(np.arange(len(protein_order)))
    ax_protein.set_xticklabels(protein_order, rotation=0)
    ax_protein.set_title("By protein", fontsize=12.5, pad=10)
    ax_protein.set_xlabel("Protein")
    ax_protein.set_ylabel("Skipped windows")
    _style_axes(ax_protein, axis="y")
    for bar, value in zip(protein_bars, protein_counts):
        ax_protein.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.3 if value > 0 else 0.1,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#38424B",
        )

    if cluster_summary_df.empty:
        ax_cluster.text(
            0.5,
            0.5,
            "No skipped windows",
            ha="center",
            va="center",
            fontsize=12,
            color=neutral_dark,
            transform=ax_cluster.transAxes,
        )
        ax_cluster.set_xticks([])
        ax_cluster.set_yticks([])
        ax_cluster.set_title("By motif cluster", fontsize=12.5, pad=10)
        _style_axes(ax_cluster, show_grid=False)
    else:
        cluster_summary = cluster_summary_df.sort_values("motif_cluster_id", kind="mergesort")
        cluster_ids = cluster_summary["motif_cluster_id"].astype(str).tolist()
        cluster_counts = pd.to_numeric(
            cluster_summary["skipped_count"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        cluster_bars = ax_cluster.bar(
            np.arange(len(cluster_ids)),
            cluster_counts,
            width=0.72,
            color="#8A97A3",
            edgecolor="white",
            linewidth=0.8,
        )
        ax_cluster.set_xticks(np.arange(len(cluster_ids)))
        ax_cluster.set_xticklabels(cluster_ids, rotation=35, ha="right")
        ax_cluster.set_title("By motif cluster", fontsize=12.5, pad=10)
        ax_cluster.set_xlabel("Motif cluster")
        ax_cluster.set_ylabel("Skipped windows")
        _style_axes(ax_cluster, axis="y")
        for bar, value in zip(cluster_bars, cluster_counts):
            ax_cluster.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.3 if value > 0 else 0.1,
                f"{int(value)}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                color="#38424B",
            )

    _add_figure_header(
        fig,
        "Skipped / Ambiguous Window Distribution",
        "counts of skipped or ambiguous windows retained as validation QC context",
        title_y=0.975,
        subtitle_y=0.922,
    )
    fig.subplots_adjust(top=0.82, wspace=0.24)
    return _save_figure(
        fig,
        output_dir / "skipped_window_summary.png",
        output_dir / "skipped_window_summary.pdf",
    )


def plot_dssp_majority_fraction_by_cluster(
    window_validation_df: pd.DataFrame,
    output_dir: Path,
    random_seed: int,
    verbose: bool = False,
) -> tuple[Path, Path]:
    """Plot boxplots and jittered points for DSSP majority-fraction strength by cluster."""

    if verbose:
        print("Plotting DSSP majority-fraction by motif cluster")

    colors = get_dssp_class_colors()
    working_df = window_validation_df.copy()
    working_df["validation_eligible_for_metrics"] = _coerce_bool_series(
        working_df["validation_eligible_for_metrics"]
    )
    working_df["dssp_majority_fraction"] = pd.to_numeric(
        working_df["dssp_majority_fraction"], errors="coerce"
    )
    working_df = working_df.loc[
        (working_df["validation_eligible_for_metrics"] == True)
        & (~working_df["dssp_majority_fraction"].isna())
        & (working_df["dssp_majority_label"].astype(str).isin(VALID_DSSP_LABELS))
    ].copy()
    if working_df.empty:
        raise ValueError("No validation-eligible windows with dssp_majority_fraction were found")

    cluster_order = sorted(working_df["motif_cluster_id"].astype(str).unique().tolist())
    data_arrays = [
        working_df.loc[
            working_df["motif_cluster_id"].astype(str) == cluster_id, "dssp_majority_fraction"
        ].to_numpy(dtype=float)
        for cluster_id in cluster_order
    ]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    box = ax.boxplot(
        data_arrays,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#34424F", "linewidth": 1.4},
        whiskerprops={"color": "#9DA8B3", "linewidth": 1.0},
        capprops={"color": "#9DA8B3", "linewidth": 1.0},
        boxprops={"facecolor": "#EEF1F5", "edgecolor": "#B8C0C8", "linewidth": 1.0},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.95)

    for index, cluster_id in enumerate(cluster_order, start=1):
        subset = working_df.loc[working_df["motif_cluster_id"].astype(str) == cluster_id].copy()
        jitter = deterministic_jitter(len(subset), scale=0.10, seed=random_seed + index)
        x_values = np.full(len(subset), float(index)) + jitter
        point_colors = [
            colors[str(label)] for label in subset["dssp_majority_label"].astype(str).tolist()
        ]
        ax.scatter(
            x_values,
            subset["dssp_majority_fraction"].to_numpy(dtype=float),
            s=24,
            c=point_colors,
            alpha=0.68,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    ax.axhline(0.50, color="#BFC7CE", linewidth=1.0, linestyle="--", zorder=1)
    ax.axhline(0.75, color="#D8DEE4", linewidth=0.9, linestyle=":", zorder=1)
    ax.set_xticks(np.arange(1, len(cluster_order) + 1))
    ax.set_xticklabels(cluster_order, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Motif cluster")
    ax.set_ylabel("DSSP majority fraction")
    ax.set_title("DSSP Majority Strength by Motif Cluster", fontsize=16, fontweight="bold", pad=14)
    _style_axes(ax, axis="y")
    ax.legend(
        handles=[
            mlines.Line2D(
                [], [], marker="o", linestyle="none", markersize=6.2, markerfacecolor=colors["H"],
                markeredgecolor="white", markeredgewidth=0.5, label="H"
            ),
            mlines.Line2D(
                [], [], marker="o", linestyle="none", markersize=6.2, markerfacecolor=colors["E"],
                markeredgecolor="white", markeredgewidth=0.5, label="E"
            ),
            mlines.Line2D(
                [], [], marker="o", linestyle="none", markersize=6.2, markerfacecolor=colors["L"],
                markeredgecolor="white", markeredgewidth=0.5, label="L"
            ),
        ],
        title="Window DSSP label",
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    fig.subplots_adjust(right=0.83)
    return _save_figure(
        fig,
        output_dir / "dssp_majority_fraction_by_cluster.png",
        output_dir / "dssp_majority_fraction_by_cluster.pdf",
    )


def main() -> None:
    """Run the full validation plotting workflow."""

    args = parse_args()
    if args.perplexity <= 0:
        raise ValueError("--perplexity must be greater than 0")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than 0")
    if args.n_iter < 250:
        raise ValueError("--n-iter must be at least 250")

    _configure_matplotlib_style()

    script_dir = Path(__file__).resolve().parent
    validation_dir = _resolve_cli_path(args.validation_dir, script_dir)
    comparison_dir = _resolve_cli_path(args.comparison_dir, script_dir)
    output_dir = _resolve_cli_path(args.output_dir, script_dir)
    ensure_output_dir(output_dir)

    data = load_inputs(validation_dir, comparison_dir)
    validate_inputs(data)

    cluster_label_mapping_df = _get_dataframe(data, "cluster_label_mapping")
    cluster_purity_df = _get_dataframe(data, "cluster_purity")
    precision_recall_df = _get_dataframe(data, "precision_recall")
    window_validation_df = _get_dataframe(data, "window_validation")
    skipped_windows_df = _get_dataframe(data, "skipped_windows")
    window_features_df = _get_dataframe(data, "window_features")

    coord_df, _ = load_or_create_tsne_coordinates(
        window_features_df=window_features_df,
        comparison_dir=comparison_dir,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        random_seed=args.random_seed,
        verbose=args.verbose,
    )

    composition_df = build_per_protein_dssp_composition(window_validation_df)
    accuracy_df = build_per_protein_accuracy(window_validation_df, cluster_label_mapping_df)
    skipped_protein_df, skipped_cluster_df = build_skipped_window_summaries(skipped_windows_df)

    outputs: list[Path] = []
    for plotter in (
        lambda: plot_cluster_purity_heatmap(cluster_purity_df, output_dir, verbose=args.verbose),
        lambda: plot_cluster_dssp_composition(cluster_purity_df, output_dir, verbose=args.verbose),
        lambda: plot_tsne_by_dssp_label(coord_df, window_validation_df, output_dir, verbose=args.verbose),
        lambda: plot_class_metrics(precision_recall_df, output_dir, verbose=args.verbose),
        lambda: plot_per_protein_dssp_composition(composition_df, output_dir, verbose=args.verbose),
        lambda: plot_per_protein_validation_accuracy(accuracy_df, output_dir, verbose=args.verbose),
        lambda: plot_skipped_window_summary(
            skipped_protein_df, skipped_cluster_df, output_dir, verbose=args.verbose
        ),
        lambda: plot_dssp_majority_fraction_by_cluster(
            window_validation_df,
            output_dir,
            random_seed=args.random_seed,
            verbose=args.verbose,
        ),
    ):
        png_path, pdf_path = plotter()
        outputs.extend([png_path, pdf_path])

    for output_path in outputs:
        print(f"Output figure: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(
            "plot_validation.py was created successfully, but it could not be run because "
            f"the expected validation outputs were missing: {exc}"
        )
