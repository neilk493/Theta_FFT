"""Compare proteins and local motif candidates using precomputed window spectral features.

This module is separate from preprocessing, spectral analysis, plotting, and
validation by design. It reads precomputed sliding-window spectral features and
produces nearest-neighbor, protein-similarity, and unsupervised motif-clustering
outputs. It does not recompute FFT, create plots, or perform supervised
classification.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODULE_NAME = "protein_motif_comparison"
SCRIPT_NAME = "protein_motif_comparison.py"

METADATA_COLUMNS = [
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
    "relative_window_start_in_segment",
    "relative_window_end_in_segment",
    "relative_window_center_in_segment",
]

COMPARISON_FEATURE_COLUMNS = [
    "theta_mean_deg",
    "theta_std_deg",
    "dominant_frequency_cycles_per_residue",
    "dominant_power_fraction_nonzero",
    "spectral_centroid_cycles_per_residue",
    "spectral_bandwidth_cycles_per_residue",
    "low_band_power_fraction",
    "mid_band_power_fraction",
    "high_band_power_fraction",
    "peak1_frequency_cycles_per_residue",
    "peak1_power_fraction_nonzero",
    "peak2_frequency_cycles_per_residue",
    "peak2_power_fraction_nonzero",
    "peak3_frequency_cycles_per_residue",
    "peak3_power_fraction_nonzero",
]

REQUIRED_COLUMNS = METADATA_COLUMNS + COMPARISON_FEATURE_COLUMNS


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the protein motif comparison module."""
    parser = argparse.ArgumentParser(
        description="Compare proteins and local motif candidates using precomputed local spectral window features."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("Theta_FFT/output/spectrograms/local_window_features.csv"),
        help="Path to the precomputed local window feature table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Theta_FFT/output/comparison"),
        help="Directory for comparison outputs.",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=5,
        help="Number of nearest neighbors to report per retained window.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=6,
        help="Requested number of k-means motif clusters.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for k-means++ initialization.",
    )
    parser.add_argument(
        "--max-kmeans-iter",
        type=int,
        default=100,
        help="Maximum number of k-means iterations.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Centroid-shift tolerance for k-means convergence.",
    )
    parser.add_argument(
        "--allow-same-protein-neighbors",
        action="store_true",
        help="Allow nearest-neighbor matching within the same protein.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress details while running.",
    )
    args = parser.parse_args()

    if args.k_neighbors < 1:
        raise ValueError("k_neighbors must be >= 1")
    if args.n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    if args.max_kmeans_iter < 1:
        raise ValueError("max_kmeans_iter must be >= 1")
    if args.tolerance <= 0:
        raise ValueError("tolerance must be > 0")

    return args


def ensure_output_dir(output_dir: Path) -> None:
    """Create the output directory if it does not already exist."""
    output_dir.mkdir(parents=True, exist_ok=True)


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_project_path(path: Path) -> Path:
    """Resolve a user path robustly when run from the project root or its parent."""
    project_root = _project_root()
    path = Path(path)
    if path.is_absolute():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    parts = path.parts
    if parts and parts[0] == project_root.name:
        stripped_candidate = project_root.joinpath(*parts[1:])
        if stripped_candidate.exists():
            return stripped_candidate.resolve()
        return stripped_candidate

    return (project_root / path).resolve()


def load_window_features(input_csv: Path) -> pd.DataFrame:
    """Load the precomputed local window feature table from CSV."""
    resolved_input = _resolve_project_path(input_csv)
    if not resolved_input.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {resolved_input}. Expected precomputed local_window_features.csv."
        )
    return pd.read_csv(resolved_input)


def validate_window_features(df: pd.DataFrame, source_name: str) -> None:
    """Validate required columns, unique IDs, missing metadata, and numeric comparison features."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {source_name}: {', '.join(missing_columns)}"
        )

    if not df["window_label"].is_unique:
        raise ValueError(f"window_label must be unique in {source_name}")

    for column in ["protein", "chain", "segment_label", "window_label"]:
        if df[column].isna().any():
            raise ValueError(f"Column '{column}' contains missing values in {source_name}")

    for column in COMPARISON_FEATURE_COLUMNS:
        original = df[column]
        coerced = pd.to_numeric(original, errors="coerce")
        bad_mask = original.notna() & coerced.isna()
        if bad_mask.any():
            bad_examples = original[bad_mask].astype(str).head(5).tolist()
            raise ValueError(
                f"Feature column '{column}' contains non-numeric values in {source_name}: {bad_examples}"
            )
        df[column] = coerced


def split_retained_and_skipped_windows(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split windows into retained complete rows and skipped rows with a clear reason."""
    missing_mask = df[feature_cols].isna().any(axis=1)
    skipped_df = df.loc[
        missing_mask, ["window_label", "protein", "chain", "segment_label", "window_id"]
    ].copy()
    skipped_df["reason"] = "missing_comparison_feature"
    retained_df = df.loc[~missing_mask].copy()
    return retained_df, skipped_df


def build_standardized_feature_matrix(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, list[str]]:
    """Standardize retained feature columns and record scaling metadata."""
    raw_matrix = df[feature_cols].to_numpy(dtype=float)
    means = raw_matrix.mean(axis=0)
    stds = raw_matrix.std(axis=0, ddof=0)
    used_mask = stds > 0

    scaling_df = pd.DataFrame(
        {
            "feature_name": feature_cols,
            "raw_mean": means,
            "raw_std": stds,
            "used_for_comparison": used_mask,
            "dropped_zero_variance": ~used_mask,
        }
    )

    retained_feature_names = [feature for feature, used in zip(feature_cols, used_mask) if used]
    if not retained_feature_names:
        raise ValueError(
            "All selected comparison features had zero variance after filtering; comparison space is empty."
        )

    used_raw_matrix = raw_matrix[:, used_mask]
    z_matrix = (used_raw_matrix - means[used_mask]) / stds[used_mask]

    retained_df = df.copy()
    for index, feature_name in enumerate(retained_feature_names):
        retained_df[f"z__{feature_name}"] = z_matrix[:, index]

    return retained_df, z_matrix, scaling_df, retained_feature_names


def compute_pairwise_distances(z_matrix: np.ndarray) -> np.ndarray:
    """Compute the full symmetric pairwise Euclidean distance matrix with NumPy."""
    squared_norms = np.sum(z_matrix * z_matrix, axis=1, keepdims=True)
    squared_distances = squared_norms + squared_norms.T - 2.0 * (z_matrix @ z_matrix.T)
    squared_distances = np.maximum(squared_distances, 0.0)
    distances = np.sqrt(squared_distances)
    np.fill_diagonal(distances, 0.0)
    return distances


def build_neighbor_table(
    retained_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    k_neighbors: int,
    allow_same_protein_neighbors: bool,
) -> pd.DataFrame:
    """Build the nearest-neighbor table for each retained window under the chosen filter rule."""
    proteins = retained_df["protein"].astype(str).to_numpy()
    retained_reset = retained_df.reset_index(drop=True)

    records: list[dict[str, Any]] = []
    for query_index, query_row in retained_reset.iterrows():
        valid_mask = np.ones(len(retained_reset), dtype=bool)
        valid_mask[query_index] = False
        if not allow_same_protein_neighbors:
            valid_mask &= proteins != proteins[query_index]

        valid_indices = np.flatnonzero(valid_mask)
        if valid_indices.size == 0:
            continue

        valid_distances = distance_matrix[query_index, valid_indices]
        neighbor_order = np.argsort(valid_distances)[:k_neighbors]
        chosen_indices = valid_indices[neighbor_order]

        for rank, neighbor_index in enumerate(chosen_indices, start=1):
            neighbor_row = retained_reset.iloc[neighbor_index]
            distance = float(distance_matrix[query_index, neighbor_index])
            records.append(
                {
                    "query_window_label": query_row["window_label"],
                    "query_protein": query_row["protein"],
                    "query_chain": query_row["chain"],
                    "query_segment_label": query_row["segment_label"],
                    "query_window_id": query_row["window_id"],
                    "query_window_center_seq_index": query_row["window_center_seq_index"],
                    "query_relative_window_center_in_segment": query_row[
                        "relative_window_center_in_segment"
                    ],
                    "neighbor_rank": rank,
                    "neighbor_window_label": neighbor_row["window_label"],
                    "neighbor_protein": neighbor_row["protein"],
                    "neighbor_chain": neighbor_row["chain"],
                    "neighbor_segment_label": neighbor_row["segment_label"],
                    "neighbor_window_id": neighbor_row["window_id"],
                    "neighbor_window_center_seq_index": neighbor_row["window_center_seq_index"],
                    "neighbor_relative_window_center_in_segment": neighbor_row[
                        "relative_window_center_in_segment"
                    ],
                    "distance_euclidean_zspace": distance,
                    "similarity_score": 1.0 / (1.0 + distance),
                    "same_protein": bool(query_row["protein"] == neighbor_row["protein"]),
                    "same_chain": bool(query_row["chain"] == neighbor_row["chain"]),
                    "same_segment": bool(query_row["segment_label"] == neighbor_row["segment_label"]),
                }
            )

    neighbor_df = pd.DataFrame.from_records(records)
    if neighbor_df.empty:
        raise ValueError(
            "No valid neighbors exist under the current neighbor filter rule. "
            "Consider using --allow-same-protein-neighbors only if that is biologically intended."
        )
    return neighbor_df


def build_reciprocal_match_table(neighbor_df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique reciprocal best-match window pairs from the nearest-neighbor table."""
    best_df = neighbor_df.loc[neighbor_df["neighbor_rank"] == 1].copy()
    best_map = dict(zip(best_df["query_window_label"], best_df["neighbor_window_label"]))
    row_map = {row["query_window_label"]: row for _, row in best_df.iterrows()}

    pair_records: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for window_a, window_b in best_map.items():
        if best_map.get(window_b) != window_a:
            continue
        pair_key = tuple(sorted((window_a, window_b)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        row_a = row_map[pair_key[0]]
        if row_a["neighbor_window_label"] != pair_key[1]:
            row_a = row_map[pair_key[1]]

        pair_records.append(
            {
                "window_a_label": row_a["query_window_label"],
                "protein_a": row_a["query_protein"],
                "chain_a": row_a["query_chain"],
                "segment_label_a": row_a["query_segment_label"],
                "window_id_a": row_a["query_window_id"],
                "window_center_seq_index_a": row_a["query_window_center_seq_index"],
                "relative_window_center_in_segment_a": row_a[
                    "query_relative_window_center_in_segment"
                ],
                "window_b_label": row_a["neighbor_window_label"],
                "protein_b": row_a["neighbor_protein"],
                "chain_b": row_a["neighbor_chain"],
                "segment_label_b": row_a["neighbor_segment_label"],
                "window_id_b": row_a["neighbor_window_id"],
                "window_center_seq_index_b": row_a["neighbor_window_center_seq_index"],
                "relative_window_center_in_segment_b": row_a[
                    "neighbor_relative_window_center_in_segment"
                ],
                "distance_euclidean_zspace": row_a["distance_euclidean_zspace"],
                "similarity_score": row_a["similarity_score"],
                "same_protein": row_a["same_protein"],
                "same_chain": row_a["same_chain"],
                "same_segment": row_a["same_segment"],
            }
        )

    return pd.DataFrame.from_records(pair_records)


def build_protein_similarity_tables(
    retained_df: pd.DataFrame, distance_matrix: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build protein-level similarity summaries from minimum cross-protein window distances."""
    protein_array = retained_df["protein"].astype(str).to_numpy()
    proteins = sorted(np.unique(protein_array).tolist())
    protein_indices = {
        protein: np.flatnonzero(protein_array == protein) for protein in proteins
    }

    long_records: list[dict[str, Any]] = []
    matrix_df = pd.DataFrame(
        np.eye(len(proteins), dtype=float), index=proteins, columns=proteins
    )

    for index, protein_a in enumerate(proteins):
        for protein_b in proteins[index + 1 :]:
            idx_a = protein_indices[protein_a]
            idx_b = protein_indices[protein_b]
            distances_ab = distance_matrix[np.ix_(idx_a, idx_b)]
            mean_min_distance_a_to_b = float(np.min(distances_ab, axis=1).mean())
            mean_min_distance_b_to_a = float(np.min(distances_ab, axis=0).mean())
            symmetric_distance = 0.5 * (
                mean_min_distance_a_to_b + mean_min_distance_b_to_a
            )
            similarity_score = 1.0 / (1.0 + symmetric_distance)

            long_records.append(
                {
                    "protein_a": protein_a,
                    "protein_b": protein_b,
                    "n_windows_a": int(idx_a.size),
                    "n_windows_b": int(idx_b.size),
                    "mean_min_distance_a_to_b": mean_min_distance_a_to_b,
                    "mean_min_distance_b_to_a": mean_min_distance_b_to_a,
                    "symmetric_distance": symmetric_distance,
                    "similarity_score": similarity_score,
                }
            )
            matrix_df.loc[protein_a, protein_b] = similarity_score
            matrix_df.loc[protein_b, protein_a] = similarity_score

    return pd.DataFrame.from_records(long_records), matrix_df


def kmeans_plus_plus_init(z_matrix: np.ndarray, n_clusters: int, random_seed: int) -> np.ndarray:
    """Initialize k-means centroids with k-means++ using Euclidean squared distance."""
    n_samples = z_matrix.shape[0]
    rng = np.random.default_rng(random_seed)
    centroids = np.empty((n_clusters, z_matrix.shape[1]), dtype=float)

    first_index = int(rng.integers(0, n_samples))
    centroids[0] = z_matrix[first_index]
    closest_squared_dist = np.sum((z_matrix - centroids[0]) ** 2, axis=1)

    for centroid_index in range(1, n_clusters):
        total = float(closest_squared_dist.sum())
        if total <= 0:
            next_index = int(rng.integers(0, n_samples))
        else:
            probabilities = closest_squared_dist / total
            next_index = int(rng.choice(n_samples, p=probabilities))
        centroids[centroid_index] = z_matrix[next_index]
        candidate_squared_dist = np.sum((z_matrix - centroids[centroid_index]) ** 2, axis=1)
        closest_squared_dist = np.minimum(closest_squared_dist, candidate_squared_dist)

    return centroids


def run_kmeans(
    z_matrix: np.ndarray,
    n_clusters: int,
    random_seed: int,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Run NumPy k-means with k-means++, centroid updates, and empty-cluster recovery."""
    centroids = kmeans_plus_plus_init(z_matrix, n_clusters, random_seed)
    labels = np.zeros(z_matrix.shape[0], dtype=int)
    iteration = 0

    for iteration in range(1, max_iter + 1):
        distances = np.linalg.norm(z_matrix[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(z_matrix.shape[0]), labels].copy()

        new_centroids = centroids.copy()
        for cluster_index in range(n_clusters):
            member_mask = labels == cluster_index
            if np.any(member_mask):
                new_centroids[cluster_index] = z_matrix[member_mask].mean(axis=0)
            else:
                farthest_point_index = int(np.argmax(min_distances))
                new_centroids[cluster_index] = z_matrix[farthest_point_index]
                labels[farthest_point_index] = cluster_index
                min_distances[farthest_point_index] = 0.0

        centroid_shift = float(np.max(np.linalg.norm(new_centroids - centroids, axis=1)))
        centroids = new_centroids
        if centroid_shift < tolerance:
            break

    final_distances = np.linalg.norm(z_matrix[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(final_distances, axis=1)
    final_min_distances = final_distances[np.arange(z_matrix.shape[0]), labels]
    inertia = float(np.sum(final_min_distances ** 2))
    return labels, centroids, iteration, inertia


def relabel_clusters_by_size(labels: np.ndarray) -> np.ndarray:
    """Relabel clusters by descending cluster size for stable motif IDs."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    sort_order = np.lexsort((unique_labels, -counts))
    relabel_map = {
        int(original_label): int(new_label)
        for new_label, original_label in enumerate(unique_labels[sort_order])
    }
    return np.array([relabel_map[int(label)] for label in labels], dtype=int)


def build_cluster_outputs(
    retained_df: pd.DataFrame,
    z_matrix: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    retained_feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build cluster assignment, summary, and protein-composition outputs."""
    relabeled = relabel_clusters_by_size(labels)
    old_to_new: dict[int, int] = {}
    for old_label, new_label in zip(labels, relabeled):
        old_to_new.setdefault(int(old_label), int(new_label))

    reordered_centroids = np.empty_like(centroids)
    for old_label, new_label in old_to_new.items():
        reordered_centroids[new_label] = centroids[old_label]

    cluster_sizes = pd.Series(relabeled).value_counts().sort_index().to_dict()
    assigned_centroids = reordered_centroids[relabeled]
    distance_to_centroid = np.linalg.norm(z_matrix - assigned_centroids, axis=1)

    assignments_df = retained_df[
        [
            "window_label",
            "protein",
            "chain",
            "segment_label",
            "segment_id",
            "window_id",
            "window_center_seq_index",
            "relative_window_center_in_segment",
            "window_length",
            *COMPARISON_FEATURE_COLUMNS,
        ]
    ].copy()
    assignments_df["motif_cluster_id"] = [
        f"motif{cluster_id:03d}" for cluster_id in relabeled
    ]
    assignments_df["cluster_size"] = [int(cluster_sizes[int(label)]) for label in relabeled]
    assignments_df["distance_to_centroid"] = distance_to_centroid
    assignments_df["similarity_to_centroid"] = 1.0 / (1.0 + distance_to_centroid)

    summary_records: list[dict[str, Any]] = []
    protein_count_records: list[dict[str, Any]] = []

    for cluster_id in range(reordered_centroids.shape[0]):
        cluster_mask = relabeled == cluster_id
        cluster_df = retained_df.loc[cluster_mask].copy()
        cluster_z = z_matrix[cluster_mask]
        cluster_centroid = reordered_centroids[cluster_id]
        cluster_distances = np.linalg.norm(cluster_z - cluster_centroid, axis=1)
        medoid_position = int(np.argmin(cluster_distances))
        representative_row = cluster_df.iloc[medoid_position]

        summary_record = {
            "motif_cluster_id": f"motif{cluster_id:03d}",
            "n_windows": int(cluster_mask.sum()),
            "n_proteins": int(cluster_df["protein"].nunique()),
            "n_chains": int(cluster_df["chain"].nunique()),
            "mean_distance_to_centroid": float(cluster_distances.mean()),
            "representative_window_label": representative_row["window_label"],
            "representative_protein": representative_row["protein"],
            "representative_chain": representative_row["chain"],
            "representative_segment_label": representative_row["segment_label"],
            "representative_window_center_seq_index": representative_row[
                "window_center_seq_index"
            ],
        }
        raw_centroids = cluster_df[retained_feature_names].mean(axis=0)
        for feature_name in retained_feature_names:
            summary_record[f"centroid_raw__{feature_name}"] = float(raw_centroids[feature_name])
        summary_records.append(summary_record)

        protein_counts = cluster_df["protein"].value_counts().sort_index()
        cluster_size = int(cluster_mask.sum())
        for protein, count in protein_counts.items():
            protein_count_records.append(
                {
                    "motif_cluster_id": f"motif{cluster_id:03d}",
                    "protein": protein,
                    "n_windows": int(count),
                    "fraction_within_cluster": float(count / cluster_size),
                }
            )

    return (
        assignments_df,
        pd.DataFrame.from_records(summary_records),
        pd.DataFrame.from_records(protein_count_records),
    )


def write_outputs(
    output_dir: Path,
    window_features_standardized_df: pd.DataFrame,
    feature_scaling_df: pd.DataFrame,
    skipped_windows_df: pd.DataFrame,
    neighbor_df: pd.DataFrame,
    reciprocal_matches_df: pd.DataFrame,
    protein_similarity_long_df: pd.DataFrame,
    protein_similarity_matrix_df: pd.DataFrame,
    motif_cluster_assignments_df: pd.DataFrame,
    motif_cluster_summary_df: pd.DataFrame,
    motif_cluster_protein_counts_df: pd.DataFrame,
    run_metadata: dict[str, Any],
) -> dict[str, Path]:
    """Write all required CSV and JSON outputs and return their resolved paths."""
    ensure_output_dir(output_dir)

    output_paths = {
        "window_features_standardized": output_dir / "window_features_standardized.csv",
        "feature_scaling": output_dir / "feature_scaling.csv",
        "skipped_windows": output_dir / "skipped_windows.csv",
        "window_nearest_neighbors": output_dir / "window_nearest_neighbors.csv",
        "reciprocal_motif_matches": output_dir / "reciprocal_motif_matches.csv",
        "protein_similarity_long": output_dir / "protein_similarity_long.csv",
        "protein_similarity_matrix": output_dir / "protein_similarity_matrix.csv",
        "motif_cluster_assignments": output_dir / "motif_cluster_assignments.csv",
        "motif_cluster_summary": output_dir / "motif_cluster_summary.csv",
        "motif_cluster_protein_counts": output_dir / "motif_cluster_protein_counts.csv",
        "run_metadata": output_dir / "run_metadata.json",
    }

    window_features_standardized_df.to_csv(output_paths["window_features_standardized"], index=False)
    feature_scaling_df.to_csv(output_paths["feature_scaling"], index=False)
    skipped_windows_df.to_csv(output_paths["skipped_windows"], index=False)
    neighbor_df.to_csv(output_paths["window_nearest_neighbors"], index=False)
    reciprocal_matches_df.to_csv(output_paths["reciprocal_motif_matches"], index=False)
    protein_similarity_long_df.to_csv(output_paths["protein_similarity_long"], index=False)
    protein_similarity_matrix_df.to_csv(output_paths["protein_similarity_matrix"], index=True)
    motif_cluster_assignments_df.to_csv(output_paths["motif_cluster_assignments"], index=False)
    motif_cluster_summary_df.to_csv(output_paths["motif_cluster_summary"], index=False)
    motif_cluster_protein_counts_df.to_csv(output_paths["motif_cluster_protein_counts"], index=False)
    output_paths["run_metadata"].write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    return output_paths


def _verbose_print(enabled: bool, message: str) -> None:
    if enabled:
        print(message)


def main() -> None:
    """Run the local window comparison pipeline and write all required outputs."""
    args = parse_args()
    input_csv = _resolve_project_path(args.input_csv)
    output_dir = _resolve_project_path(args.output_dir)

    _verbose_print(args.verbose, "Loading window feature table...")
    print(f"Input file: {input_csv}")

    df = load_window_features(input_csv)
    validate_window_features(df, str(input_csv))

    _verbose_print(args.verbose, "Filtering windows with missing comparison features...")
    retained_df, skipped_df = split_retained_and_skipped_windows(df, COMPARISON_FEATURE_COLUMNS)
    if len(retained_df) < 2:
        raise ValueError(
            "Fewer than 2 windows remain after filtering missing comparison features."
        )

    _verbose_print(args.verbose, "Standardizing comparison features...")
    retained_with_z_df, z_matrix, feature_scaling_df, retained_feature_names = (
        build_standardized_feature_matrix(retained_df, COMPARISON_FEATURE_COLUMNS)
    )

    _verbose_print(args.verbose, "Computing pairwise distances...")
    distance_matrix = compute_pairwise_distances(z_matrix)

    _verbose_print(args.verbose, "Building nearest-neighbor comparisons...")
    neighbor_df = build_neighbor_table(
        retained_with_z_df,
        distance_matrix,
        k_neighbors=args.k_neighbors,
        allow_same_protein_neighbors=args.allow_same_protein_neighbors,
    )

    _verbose_print(args.verbose, "Finding reciprocal motif candidates...")
    reciprocal_matches_df = build_reciprocal_match_table(neighbor_df)

    _verbose_print(args.verbose, "Summarizing protein-to-protein similarity...")
    protein_similarity_long_df, protein_similarity_matrix_df = build_protein_similarity_tables(
        retained_with_z_df, distance_matrix
    )

    actual_n_clusters = min(args.n_clusters, len(retained_with_z_df))
    _verbose_print(args.verbose, "Running unsupervised motif clustering...")
    labels, centroids, n_iterations, inertia = run_kmeans(
        z_matrix,
        n_clusters=actual_n_clusters,
        random_seed=args.random_seed,
        max_iter=args.max_kmeans_iter,
        tolerance=args.tolerance,
    )

    motif_cluster_assignments_df, motif_cluster_summary_df, motif_cluster_protein_counts_df = (
        build_cluster_outputs(
            retained_with_z_df,
            z_matrix,
            labels,
            centroids,
            retained_feature_names,
        )
    )

    run_metadata = {
        "module_name": MODULE_NAME,
        "script_name": SCRIPT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "selected_comparison_features": COMPARISON_FEATURE_COLUMNS,
        "retained_features_after_zero_variance_filter": retained_feature_names,
        "dropped_zero_variance_features": feature_scaling_df.loc[
            feature_scaling_df["dropped_zero_variance"], "feature_name"
        ].tolist(),
        "n_input_windows": int(len(df)),
        "n_retained_windows": int(len(retained_with_z_df)),
        "n_skipped_windows": int(len(skipped_df)),
        "k_neighbors": int(args.k_neighbors),
        "allow_same_protein_neighbors": bool(args.allow_same_protein_neighbors),
        "requested_n_clusters": int(args.n_clusters),
        "actual_n_clusters": int(actual_n_clusters),
        "kmeans_max_iter": int(args.max_kmeans_iter),
        "kmeans_tolerance": float(args.tolerance),
        "kmeans_iterations_run": int(n_iterations),
        "kmeans_final_inertia": float(inertia),
        "n_proteins": int(retained_with_z_df["protein"].nunique()),
        "notes": [
            "input = precomputed local_window_features.csv",
            "no raw theta signals were used here",
            "no FFT was recomputed here",
            "comparison was performed in standardized window-feature space",
            "nearest-neighbor motif candidates are feature-space similarities, not confirmed structural identities",
            "clustering is unsupervised and exploratory",
            "no plotting performed in this module",
            "no DSSP validation performed in this module",
        ],
    }

    _verbose_print(args.verbose, "Writing outputs...")
    output_paths = write_outputs(
        output_dir=output_dir,
        window_features_standardized_df=retained_with_z_df,
        feature_scaling_df=feature_scaling_df,
        skipped_windows_df=skipped_df,
        neighbor_df=neighbor_df,
        reciprocal_matches_df=reciprocal_matches_df,
        protein_similarity_long_df=protein_similarity_long_df,
        protein_similarity_matrix_df=protein_similarity_matrix_df,
        motif_cluster_assignments_df=motif_cluster_assignments_df,
        motif_cluster_summary_df=motif_cluster_summary_df,
        motif_cluster_protein_counts_df=motif_cluster_protein_counts_df,
        run_metadata=run_metadata,
    )

    print(f"Input windows: {len(df)}")
    print(f"Retained windows: {len(retained_with_z_df)}")
    print(f"Skipped windows: {len(skipped_df)}")
    print(f"Proteins represented: {retained_with_z_df['protein'].nunique()}")
    print(f"Reciprocal motif matches found: {len(reciprocal_matches_df)}")
    print(f"Cluster count used: {actual_n_clusters}")
    print("Main output files:")
    for key in [
        "window_features_standardized",
        "window_nearest_neighbors",
        "reciprocal_motif_matches",
        "protein_similarity_long",
        "protein_similarity_matrix",
        "motif_cluster_assignments",
        "motif_cluster_summary",
        "run_metadata",
    ]:
        print(f"  {output_paths[key]}")


if __name__ == "__main__":
    main()
