"""Validate precomputed motif-cluster assignments against DSSP labels.

This module is the final validation stage of Theta_FFT. It is intentionally
separate from FFT analysis, comparison, clustering, and plotting. The script
maps precomputed theta windows to residue-level DSSP labels through fft_data
seq_index-to-residue bridges, then computes window-level validation records,
cluster purity, confusion counts, and precision/recall tables.

It does not recompute FFT, comparison distances, or clustering, and it does
not generate plots.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB import DSSP, PDBParser


REQUIRED_MOTIF_COLUMNS = [
    "window_label",
    "protein",
    "chain",
    "segment_label",
    "segment_id",
    "window_id",
    "window_start_seq_index",
    "window_end_seq_index",
    "window_center_seq_index",
    "window_length",
    "motif_cluster_id",
]

REQUIRED_FFT_COLUMNS = [
    "protein",
    "chain",
    "seq_index",
    "res_i",
    "res_j",
]

VALID_DSSP_LABELS = ["H", "E", "L"]
STATUS_VALIDATED = "VALIDATED"
STATUS_TIE = "TIE_MAJORITY"
STATUS_UNASSIGNED = "UNASSIGNED_NO_MAPPED_RESIDUES"
SKIPPED_WINDOWS_COLUMNS = ["window_label", "protein", "chain", "motif_cluster_id", "reason"]
SKIPPED_PROTEINS_COLUMNS = ["protein", "pdb_file", "reason", "n_windows_affected"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the validation module."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate precomputed Theta_FFT motif-cluster assignments against "
            "DSSP-derived secondary structure labels."
        )
    )
    parser.add_argument(
        "--input-comparison-dir",
        type=Path,
        default=None,
        help="Directory containing motif_cluster_assignments.csv.",
    )
    parser.add_argument(
        "--input-pdb-dir",
        type=Path,
        default=None,
        help="Directory containing PDB files.",
    )
    parser.add_argument(
        "--input-fft-data-dir",
        type=Path,
        default=None,
        help="Directory containing *_fft_data.csv bridge tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for validation CSV/JSON outputs.",
    )
    parser.add_argument(
        "--dssp-executable",
        type=str,
        default=None,
        help="Optional DSSP executable path or name. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-protein progress and mapping warnings.",
    )
    return parser.parse_args()


def ensure_output_dir(output_dir: Path) -> None:
    """Create the output directory if it does not already exist."""
    output_dir.mkdir(parents=True, exist_ok=True)


def resolve_default_paths(script_path: Path) -> dict[str, Path]:
    """Resolve default input/output directories relative to the script file."""
    project_root = script_path.resolve().parent
    return {
        "input_comparison_dir": project_root / "output" / "comparison",
        "input_pdb_dir": project_root / "pdb",
        "input_fft_data_dir": project_root / "fft_data",
        "output_dir": project_root / "output" / "validation",
    }


def find_dssp_executable(requested: str | None = None) -> str | None:
    """Find the DSSP executable from an explicit request or auto-detection."""
    if requested:
        requested_path = Path(requested)
        if requested_path.exists():
            return str(requested_path)
        if shutil.which(requested):
            return requested
        return None

    for candidate in ("mkdssp", "dssp"):
        if shutil.which(candidate):
            return candidate
    return None


def load_motif_cluster_assignments(csv_path: Path) -> pd.DataFrame:
    """Load motif-cluster assignments and reconstruct window bounds if needed."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Motif-cluster assignments file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No windows found in motif-cluster assignments file: {csv_path}")

    if (
        "window_start_seq_index" not in df.columns
        or "window_end_seq_index" not in df.columns
    ):
        required_for_derivation = {"window_center_seq_index", "window_length"}
        if not required_for_derivation.issubset(df.columns):
            missing = sorted(required_for_derivation - set(df.columns))
            raise ValueError(
                "Cannot derive window_start_seq_index/window_end_seq_index because "
                f"required columns are missing: {missing}"
            )

        center = pd.to_numeric(df["window_center_seq_index"], errors="coerce")
        length = pd.to_numeric(df["window_length"], errors="coerce")
        half_span = (length - 1) / 2.0
        start = center - half_span
        end = center + half_span

        if start.isna().any() or end.isna().any():
            raise ValueError("Unable to derive window bounds due to missing numeric values.")

        if not np.allclose(start, np.round(start)) or not np.allclose(end, np.round(end)):
            raise ValueError(
                "Derived window_start_seq_index/window_end_seq_index are not integer-like."
            )

        insert_after = df.columns.get_loc("window_center_seq_index") + 1
        df.insert(insert_after, "window_start_seq_index", np.round(start).astype(int))
        df.insert(insert_after + 1, "window_end_seq_index", np.round(end).astype(int))

    return df


def validate_motif_cluster_assignments(df: pd.DataFrame, source_name: str) -> None:
    """Validate the motif-cluster assignment table."""
    missing = [column for column in REQUIRED_MOTIF_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")

    if df.empty:
        raise ValueError(f"{source_name} contains no rows.")

    if df["window_label"].isna().any():
        raise ValueError(f"{source_name} contains missing window_label values.")

    if not df["window_label"].is_unique:
        raise ValueError(f"{source_name} contains duplicate window_label values.")

    for column in ("protein", "chain", "motif_cluster_id"):
        if df[column].isna().any():
            raise ValueError(f"{source_name} contains missing {column} values.")

    for column in ("window_start_seq_index", "window_end_seq_index", "window_length"):
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"{source_name} contains non-numeric {column} values.")
        if not np.allclose(numeric, np.round(numeric)):
            raise ValueError(f"{source_name} contains non-integer-like {column} values.")

    if (df["window_start_seq_index"].astype(float) > df["window_end_seq_index"].astype(float)).any():
        raise ValueError(f"{source_name} contains windows where start_seq_index > end_seq_index.")


def discover_fft_bridge_files(input_fft_data_dir: Path) -> dict[str, Path]:
    """Discover available FFT bridge files by protein identifier."""
    if not input_fft_data_dir.exists():
        raise FileNotFoundError(f"FFT bridge directory not found: {input_fft_data_dir}")

    file_map: dict[str, Path] = {}
    for csv_path in sorted(input_fft_data_dir.glob("*_fft_data.csv")):
        protein = csv_path.name.removesuffix("_fft_data.csv")
        file_map[protein] = csv_path

    if not file_map:
        raise FileNotFoundError(f"No FFT bridge files found in: {input_fft_data_dir}")

    return file_map


def load_fft_bridge_table(csv_path: Path) -> pd.DataFrame:
    """Load a single FFT bridge CSV table."""
    if not csv_path.exists():
        raise FileNotFoundError(f"FFT bridge file not found: {csv_path}")
    return pd.read_csv(csv_path)


def validate_fft_bridge_df(df: pd.DataFrame, source_name: str) -> None:
    """Validate an FFT bridge table used for seq_index-to-residue mapping."""
    missing = [column for column in REQUIRED_FFT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")

    if df.empty:
        raise ValueError(f"{source_name} contains no rows.")

    for column in ("protein", "chain"):
        if df[column].isna().any():
            raise ValueError(f"{source_name} contains missing {column} values.")

    for column in ("seq_index", "res_i", "res_j"):
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"{source_name} contains non-numeric {column} values.")
        if not np.allclose(numeric, np.round(numeric)):
            raise ValueError(f"{source_name} contains non-integer-like {column} values.")

    for (protein, chain), chain_df in df.groupby(["protein", "chain"], dropna=False):
        seq_index = pd.to_numeric(chain_df["seq_index"], errors="raise").astype(int)
        if seq_index.duplicated().any():
            raise ValueError(
                f"{source_name} contains duplicate seq_index values for protein={protein}, chain={chain}."
            )
        seq_sorted = np.sort(seq_index.to_numpy())
        diffs = np.diff(seq_sorted)
        if len(diffs) and np.any(diffs != 1):
            raise ValueError(
                f"{source_name} seq_index is not consecutive for protein={protein}, chain={chain}."
            )


def build_fft_bridge_maps(fft_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build per-chain FFT bridge tables for a single protein."""
    bridge_maps: dict[str, pd.DataFrame] = {}
    for chain, chain_df in fft_df.groupby("chain", dropna=False):
        prepared = chain_df.copy()
        for column in ("seq_index", "res_i", "res_j"):
            prepared[column] = pd.to_numeric(prepared[column], errors="raise").astype(int)
        prepared = prepared.sort_values("seq_index").reset_index(drop=True)
        bridge_maps[str(chain)] = prepared
    return bridge_maps


def find_pdb_file_for_protein(protein: str, input_pdb_dir: Path) -> Path | None:
    """Locate the PDB file for a protein identifier."""
    candidates = [
        input_pdb_dir / f"{protein}.pdb",
        input_pdb_dir / f"{protein.lower()}.pdb",
        input_pdb_dir / f"{protein.upper()}.pdb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(input_pdb_dir.glob(f"{protein}*.pdb"))
    return matches[0] if matches else None


def simplify_dssp_label(raw_code: str) -> str:
    """Collapse raw DSSP secondary structure codes into H/E/L classes."""
    code = "" if raw_code is None else str(raw_code).strip().upper()
    if code in {"H", "G", "I", "P"}:
        return "H"
    if code in {"E", "B"}:
        return "E"
    return "L"


def run_dssp_on_pdb(pdb_path: Path, dssp_executable: str, verbose: bool = False) -> pd.DataFrame:
    """Run DSSP on a PDB file and return residue-level labels as a DataFrame."""
    if verbose:
        print(f"[verbose] Running DSSP on {pdb_path.name} with executable '{dssp_executable}'.")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = next(structure.get_models())
    dssp = DSSP(model, str(pdb_path), dssp=dssp_executable)

    rows: list[dict[str, Any]] = []
    for key in dssp.keys():
        chain_id, residue_id = key
        _, residue_number, insertion_code = residue_id
        dssp_record = dssp[key]
        amino_acid = dssp_record[1]
        dssp_raw_code = dssp_record[2]
        rows.append(
            {
                "chain": str(chain_id),
                "residue_number": int(residue_number),
                "insertion_code": str(insertion_code).strip(),
                "amino_acid": amino_acid,
                "dssp_raw_code": dssp_raw_code,
                "dssp_simple_label": simplify_dssp_label(str(dssp_raw_code)),
            }
        )

    return pd.DataFrame(rows)


def collapse_dssp_residue_labels(dssp_df: pd.DataFrame) -> dict[tuple[str, int], str | None]:
    """Collapse DSSP labels to unique (chain, residue_number) mappings."""
    residue_label_map: dict[tuple[str, int], str | None] = {}
    if dssp_df.empty:
        return residue_label_map

    for (chain, residue_number), group_df in dssp_df.groupby(["chain", "residue_number"], dropna=False):
        labels = [str(label) for label in group_df["dssp_simple_label"].dropna().tolist()]
        unique_labels = sorted(set(labels))
        if len(unique_labels) <= 1:
            residue_label_map[(str(chain), int(residue_number))] = unique_labels[0] if unique_labels else None
        else:
            residue_label_map[(str(chain), int(residue_number))] = None
    return residue_label_map


def derive_window_residue_numbers(window_row: pd.Series, bridge_df: pd.DataFrame) -> list[int]:
    """Derive the ordered PDB residue numbers touched by a window."""
    start = int(window_row["window_start_seq_index"])
    end = int(window_row["window_end_seq_index"])
    expected_rows = end - start + 1

    subset = bridge_df.loc[
        (bridge_df["seq_index"] >= start) & (bridge_df["seq_index"] <= end)
    ].sort_values("seq_index")

    if subset.empty:
        raise ValueError("No FFT bridge rows found for the requested window range.")
    if len(subset) != expected_rows:
        raise ValueError(
            f"Window expected {expected_rows} seq_index rows but found {len(subset)}."
        )

    if "window_length" in window_row and not pd.isna(window_row["window_length"]):
        window_length = int(round(float(window_row["window_length"])))
        if expected_rows != window_length:
            raise ValueError(
                f"Window length mismatch: expected_rows={expected_rows}, window_length={window_length}."
            )

    residues = [int(subset.iloc[0]["res_i"])] + [int(value) for value in subset["res_j"].tolist()]
    ordered_unique: list[int] = []
    seen: set[int] = set()
    for residue in residues:
        if residue not in seen:
            seen.add(residue)
            ordered_unique.append(residue)
    return ordered_unique


def assign_window_dssp_labels(
    window_row: pd.Series,
    bridge_df: pd.DataFrame,
    residue_label_map: dict[tuple[str, int], str | None],
    pdb_file_name: str,
    dssp_run_status: str,
) -> dict[str, Any]:
    """Assign simplified DSSP labels to a window and compute voting statistics."""
    default_result: dict[str, Any] = {
        "validation_status": "SKIPPED_WINDOW_MAPPING_FAILED",
        "pdb_file_used": pdb_file_name,
        "dssp_run_status": dssp_run_status,
        "window_residue_numbers_pdb": np.nan,
        "window_residue_dssp_pairs": np.nan,
        "n_window_residues_total": 0,
        "n_window_residues_mapped_to_dssp": 0,
        "n_window_residues_unmapped": 0,
        "dssp_count_H": 0,
        "dssp_count_E": 0,
        "dssp_count_L": 0,
        "dssp_majority_label": "UNASSIGNED",
        "dssp_majority_fraction": np.nan,
        "validation_eligible_for_metrics": False,
    }

    try:
        residue_numbers = derive_window_residue_numbers(window_row, bridge_df)
    except Exception:
        return default_result

    chain = str(window_row["chain"])
    mapped_pairs: list[str] = []
    mapped_labels: list[str] = []
    for residue_number in residue_numbers:
        mapped_label = residue_label_map.get((chain, residue_number))
        if mapped_label in VALID_DSSP_LABELS:
            mapped_pairs.append(f"{residue_number}:{mapped_label}")
            mapped_labels.append(mapped_label)

    counts = {label: mapped_labels.count(label) for label in VALID_DSSP_LABELS}
    n_total = len(residue_numbers)
    n_mapped = len(mapped_labels)
    n_unmapped = n_total - n_mapped

    result = {
        "validation_status": STATUS_UNASSIGNED,
        "pdb_file_used": pdb_file_name,
        "dssp_run_status": dssp_run_status,
        "window_residue_numbers_pdb": ";".join(str(value) for value in residue_numbers),
        "window_residue_dssp_pairs": ";".join(mapped_pairs) if mapped_pairs else np.nan,
        "n_window_residues_total": n_total,
        "n_window_residues_mapped_to_dssp": n_mapped,
        "n_window_residues_unmapped": n_unmapped,
        "dssp_count_H": counts["H"],
        "dssp_count_E": counts["E"],
        "dssp_count_L": counts["L"],
        "dssp_majority_label": "UNASSIGNED",
        "dssp_majority_fraction": np.nan,
        "validation_eligible_for_metrics": False,
    }

    if n_mapped == 0:
        return result

    max_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == max_count]
    if len(winners) != 1:
        result["validation_status"] = STATUS_TIE
        result["dssp_majority_label"] = "TIE"
        return result

    majority_label = winners[0]
    result["validation_status"] = STATUS_VALIDATED
    result["dssp_majority_label"] = majority_label
    result["dssp_majority_fraction"] = max_count / n_mapped
    result["validation_eligible_for_metrics"] = True
    return result


def process_protein_windows(
    protein: str,
    protein_windows_df: pd.DataFrame,
    input_pdb_dir: Path,
    fft_bridge_file_map: dict[str, Path],
    dssp_executable: str | None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process all windows for one protein and return validation/skip tables."""
    if verbose:
        print(f"[verbose] Processing protein {protein} with {len(protein_windows_df)} windows.")

    window_records: list[dict[str, Any]] = []
    skipped_windows_records: list[dict[str, Any]] = []
    skipped_proteins_records: list[dict[str, Any]] = []

    pdb_path = find_pdb_file_for_protein(protein, input_pdb_dir)
    fft_bridge_path = fft_bridge_file_map.get(protein)

    protein_failure_reason: str | None = None
    protein_dssp_status = "OK"
    residue_label_map: dict[tuple[str, int], str | None] = {}
    bridge_maps: dict[str, pd.DataFrame] = {}

    if fft_bridge_path is None:
        protein_failure_reason = "fft_bridge_not_found"
        protein_dssp_status = "FFT_BRIDGE_NOT_FOUND"
        if verbose:
            print(f"[verbose] FFT bridge file missing for protein {protein}.")
    else:
        fft_df = load_fft_bridge_table(fft_bridge_path)
        validate_fft_bridge_df(fft_df, fft_bridge_path.name)
        bridge_maps = build_fft_bridge_maps(fft_df)

    if protein_failure_reason is None and pdb_path is None:
        protein_failure_reason = "pdb_file_not_found"
        protein_dssp_status = "PDB_NOT_FOUND"
        if verbose:
            print(f"[verbose] PDB file missing for protein {protein}.")

    if protein_failure_reason is None and dssp_executable is None:
        protein_failure_reason = "dssp_executable_not_found"
        protein_dssp_status = "NOT_FOUND"
        if verbose:
            print(f"[verbose] DSSP executable unavailable for protein {protein}.")

    if protein_failure_reason is None and pdb_path is not None and dssp_executable is not None:
        try:
            dssp_df = run_dssp_on_pdb(pdb_path, dssp_executable, verbose=verbose)
            residue_label_map = collapse_dssp_residue_labels(dssp_df)
            ambiguous_count = sum(value is None for value in residue_label_map.values())
            if verbose and ambiguous_count:
                print(
                    f"[verbose] Protein {protein} has {ambiguous_count} ambiguous "
                    "residue-number mappings after insertion-code collapse."
                )
        except Exception as exc:
            protein_failure_reason = "dssp_failed"
            protein_dssp_status = "FAILED"
            if verbose:
                print(f"[verbose] DSSP failed for protein {protein}: {exc}")

    if protein_failure_reason is not None:
        skipped_proteins_records.append(
            {
                "protein": protein,
                "pdb_file": pdb_path.name if pdb_path else np.nan,
                "reason": protein_failure_reason,
                "n_windows_affected": int(len(protein_windows_df)),
            }
        )

        status_map = {
            "fft_bridge_not_found": "SKIPPED_NO_FFT_BRIDGE",
            "pdb_file_not_found": "SKIPPED_NO_PDB",
            "dssp_executable_not_found": "SKIPPED_DSSP_FAILED",
            "dssp_failed": "SKIPPED_DSSP_FAILED",
        }
        status = status_map[protein_failure_reason]
        for _, row in protein_windows_df.iterrows():
            record = row.to_dict()
            record.update(
                {
                    "validation_status": status,
                    "pdb_file_used": pdb_path.name if pdb_path else np.nan,
                    "dssp_run_status": protein_dssp_status,
                    "window_residue_numbers_pdb": np.nan,
                    "window_residue_dssp_pairs": np.nan,
                    "n_window_residues_total": 0,
                    "n_window_residues_mapped_to_dssp": 0,
                    "n_window_residues_unmapped": 0,
                    "dssp_count_H": 0,
                    "dssp_count_E": 0,
                    "dssp_count_L": 0,
                    "dssp_majority_label": "UNASSIGNED",
                    "dssp_majority_fraction": np.nan,
                    "validation_eligible_for_metrics": False,
                }
            )
            window_records.append(record)
            skipped_windows_records.append(
                {
                    "window_label": row["window_label"],
                    "protein": row["protein"],
                    "chain": row["chain"],
                    "motif_cluster_id": row["motif_cluster_id"],
                    "reason": status,
                }
            )

        return (
            pd.DataFrame(window_records),
            pd.DataFrame(skipped_windows_records),
            pd.DataFrame(skipped_proteins_records),
        )

    dssp_chains = {chain for chain, _ in residue_label_map.keys()}

    for _, row in protein_windows_df.iterrows():
        record = row.to_dict()
        chain = str(row["chain"])

        if chain not in bridge_maps:
            validation_result = {
                "validation_status": "SKIPPED_CHAIN_NOT_IN_FFT_BRIDGE",
                "pdb_file_used": pdb_path.name if pdb_path else np.nan,
                "dssp_run_status": protein_dssp_status,
                "window_residue_numbers_pdb": np.nan,
                "window_residue_dssp_pairs": np.nan,
                "n_window_residues_total": 0,
                "n_window_residues_mapped_to_dssp": 0,
                "n_window_residues_unmapped": 0,
                "dssp_count_H": 0,
                "dssp_count_E": 0,
                "dssp_count_L": 0,
                "dssp_majority_label": "UNASSIGNED",
                "dssp_majority_fraction": np.nan,
                "validation_eligible_for_metrics": False,
            }
            if verbose:
                print(f"[verbose] Protein {protein}, chain {chain} missing from FFT bridge.")
        elif chain not in dssp_chains:
            validation_result = {
                "validation_status": "SKIPPED_CHAIN_NOT_IN_DSSP",
                "pdb_file_used": pdb_path.name if pdb_path else np.nan,
                "dssp_run_status": protein_dssp_status,
                "window_residue_numbers_pdb": np.nan,
                "window_residue_dssp_pairs": np.nan,
                "n_window_residues_total": 0,
                "n_window_residues_mapped_to_dssp": 0,
                "n_window_residues_unmapped": 0,
                "dssp_count_H": 0,
                "dssp_count_E": 0,
                "dssp_count_L": 0,
                "dssp_majority_label": "UNASSIGNED",
                "dssp_majority_fraction": np.nan,
                "validation_eligible_for_metrics": False,
            }
            if verbose:
                print(f"[verbose] Protein {protein}, chain {chain} missing from DSSP output.")
        else:
            validation_result = assign_window_dssp_labels(
                window_row=row,
                bridge_df=bridge_maps[chain],
                residue_label_map=residue_label_map,
                pdb_file_name=pdb_path.name if pdb_path else np.nan,
                dssp_run_status=protein_dssp_status,
            )

        record.update(validation_result)
        window_records.append(record)

        if validation_result["validation_status"] != STATUS_VALIDATED:
            skipped_windows_records.append(
                {
                    "window_label": row["window_label"],
                    "protein": row["protein"],
                    "chain": row["chain"],
                    "motif_cluster_id": row["motif_cluster_id"],
                    "reason": validation_result["validation_status"],
                }
            )

    return (
        pd.DataFrame(window_records),
        pd.DataFrame(skipped_windows_records),
        pd.DataFrame(skipped_proteins_records),
    )


def build_confusion_matrix(window_validation_df: pd.DataFrame) -> pd.DataFrame:
    """Build the cluster-by-DSSP-label confusion count table."""
    all_clusters = sorted(window_validation_df["motif_cluster_id"].astype(str).unique().tolist())
    eligible_df = window_validation_df.loc[
        window_validation_df["validation_eligible_for_metrics"] == True
    ].copy()

    if eligible_df.empty:
        confusion_df = pd.DataFrame(0, index=all_clusters, columns=VALID_DSSP_LABELS)
    else:
        confusion_df = pd.crosstab(
            eligible_df["motif_cluster_id"].astype(str),
            eligible_df["dssp_majority_label"].astype(str),
            dropna=False,
        )
        confusion_df = confusion_df.reindex(index=all_clusters, fill_value=0)
        confusion_df = confusion_df.reindex(columns=VALID_DSSP_LABELS, fill_value=0)

    confusion_df.index.name = "motif_cluster_id"
    return confusion_df.reset_index()


def build_cluster_label_mapping(
    window_validation_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map each motif cluster to a predicted structural class."""
    total_counts = (
        window_validation_df.groupby("motif_cluster_id", dropna=False)
        .size()
        .rename("n_total_windows_in_cluster")
    )
    validated_counts = (
        window_validation_df.loc[window_validation_df["validation_eligible_for_metrics"] == True]
        .groupby("motif_cluster_id", dropna=False)
        .size()
        .rename("n_validated_windows_in_cluster")
    )

    rows: list[dict[str, Any]] = []
    for _, row in confusion_df.sort_values("motif_cluster_id").iterrows():
        cluster = str(row["motif_cluster_id"])
        counts = {label: int(row[label]) for label in VALID_DSSP_LABELS}
        n_validated = int(validated_counts.get(cluster, 0))
        n_total = int(total_counts.get(cluster, 0))
        max_count = max(counts.values())
        winners = [label for label, count in counts.items() if count == max_count and count > 0]

        if n_validated == 0:
            predicted = "UNASSIGNED"
            majority_fraction = np.nan
            mapping_status = "NO_VALIDATED_WINDOWS"
        elif len(winners) == 1:
            predicted = winners[0]
            majority_fraction = max_count / n_validated
            mapping_status = "RESOLVED"
        else:
            predicted = "UNRESOLVED_TIE"
            majority_fraction = np.nan
            mapping_status = "TIE"

        rows.append(
            {
                "motif_cluster_id": cluster,
                "predicted_structural_class": predicted,
                "n_total_windows_in_cluster": n_total,
                "n_validated_windows_in_cluster": n_validated,
                "count_H": counts["H"],
                "count_E": counts["E"],
                "count_L": counts["L"],
                "majority_fraction": majority_fraction,
                "mapping_status": mapping_status,
            }
        )

    return pd.DataFrame(rows)


def build_cluster_purity(
    window_validation_df: pd.DataFrame,
    cluster_label_mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build cluster purity rows for every cluster x H/E/L combination."""
    total_counts = (
        window_validation_df.groupby("motif_cluster_id", dropna=False)
        .size()
        .rename("n_total_windows_in_cluster")
    )
    validated_df = window_validation_df.loc[
        window_validation_df["validation_eligible_for_metrics"] == True
    ].copy()
    validated_counts = (
        validated_df.groupby("motif_cluster_id", dropna=False)
        .size()
        .rename("n_validated_windows_in_cluster")
    )
    label_counts = (
        validated_df.groupby(["motif_cluster_id", "dssp_majority_label"], dropna=False)
        .size()
        .rename("n_windows_with_label")
    )

    mapping_lookup = cluster_label_mapping_df.set_index("motif_cluster_id")
    rows: list[dict[str, Any]] = []
    for cluster in sorted(window_validation_df["motif_cluster_id"].astype(str).unique().tolist()):
        n_total = int(total_counts.get(cluster, 0))
        n_validated = int(validated_counts.get(cluster, 0))
        assigned_class = mapping_lookup.loc[cluster, "predicted_structural_class"]
        cluster_majority_fraction = mapping_lookup.loc[cluster, "majority_fraction"]

        for label in VALID_DSSP_LABELS:
            n_label = int(label_counts.get((cluster, label), 0))
            rows.append(
                {
                    "motif_cluster_id": cluster,
                    "dssp_label": label,
                    "n_windows_with_label": n_label,
                    "n_validated_windows_in_cluster": n_validated,
                    "n_total_windows_in_cluster": n_total,
                    "fraction_of_validated_windows": (
                        n_label / n_validated if n_validated else np.nan
                    ),
                    "fraction_of_total_windows": n_label / n_total if n_total else np.nan,
                    "cluster_assigned_class": assigned_class,
                    "cluster_majority_fraction": cluster_majority_fraction,
                }
            )

    return pd.DataFrame(rows)


def build_precision_recall(
    window_validation_df: pd.DataFrame,
    cluster_label_mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-class precision, recall, and F1 from cluster-assigned classes."""
    scored_df = window_validation_df.loc[
        window_validation_df["validation_eligible_for_metrics"] == True
    ].merge(
        cluster_label_mapping_df[["motif_cluster_id", "predicted_structural_class"]],
        on="motif_cluster_id",
        how="left",
    )

    scored_df = scored_df.loc[
        scored_df["predicted_structural_class"].isin(VALID_DSSP_LABELS)
    ].copy()
    n_scored_windows = int(len(scored_df))

    cluster_assignment_counts = (
        cluster_label_mapping_df["predicted_structural_class"]
        .value_counts(dropna=False)
        .to_dict()
    )

    rows: list[dict[str, Any]] = []
    for structural_class in VALID_DSSP_LABELS:
        predicted_mask = scored_df["predicted_structural_class"] == structural_class
        truth_mask = scored_df["dssp_majority_label"] == structural_class

        true_positive = int((predicted_mask & truth_mask).sum())
        false_positive = int((predicted_mask & ~truth_mask).sum())
        false_negative = int((~predicted_mask & truth_mask).sum())

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else np.nan
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else np.nan
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if not (pd.isna(precision) or pd.isna(recall) or (precision + recall) == 0)
            else np.nan
        )

        rows.append(
            {
                "structural_class": structural_class,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support_ground_truth": int(truth_mask.sum()),
                "support_predicted": int(predicted_mask.sum()),
                "n_clusters_assigned_to_class": int(cluster_assignment_counts.get(structural_class, 0)),
                "n_scored_windows": n_scored_windows,
            }
        )

    return pd.DataFrame(rows)


def write_outputs(
    output_dir: Path,
    window_validation_df: pd.DataFrame,
    cluster_purity_df: pd.DataFrame,
    cluster_label_mapping_df: pd.DataFrame,
    precision_recall_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    skipped_proteins_df: pd.DataFrame,
    skipped_windows_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """Write all required CSV and JSON outputs."""
    ensure_output_dir(output_dir)

    window_validation_df.to_csv(output_dir / "window_validation.csv", index=False)
    cluster_purity_df.to_csv(output_dir / "cluster_purity.csv", index=False)
    cluster_label_mapping_df.to_csv(output_dir / "cluster_label_mapping.csv", index=False)
    precision_recall_df.to_csv(output_dir / "precision_recall_by_class.csv", index=False)
    confusion_df.to_csv(output_dir / "confusion_matrix.csv", index=False)
    skipped_proteins_df.to_csv(output_dir / "skipped_proteins.csv", index=False)
    skipped_windows_df.to_csv(output_dir / "skipped_windows.csv", index=False)

    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    """Run the validation workflow from motif-cluster assignments to CSV outputs."""
    args = parse_args()
    script_path = Path(__file__).resolve()
    defaults = resolve_default_paths(script_path)

    input_comparison_dir = (
        args.input_comparison_dir.resolve()
        if args.input_comparison_dir is not None
        else defaults["input_comparison_dir"]
    )
    input_pdb_dir = (
        args.input_pdb_dir.resolve() if args.input_pdb_dir is not None else defaults["input_pdb_dir"]
    )
    input_fft_data_dir = (
        args.input_fft_data_dir.resolve()
        if args.input_fft_data_dir is not None
        else defaults["input_fft_data_dir"]
    )
    output_dir = args.output_dir.resolve() if args.output_dir is not None else defaults["output_dir"]

    ensure_output_dir(output_dir)

    motif_cluster_assignments_file = input_comparison_dir / "motif_cluster_assignments.csv"
    dssp_executable_requested = args.dssp_executable if args.dssp_executable else "auto-detect"
    dssp_executable_used = find_dssp_executable(args.dssp_executable)

    print(f"Comparison input dir: {input_comparison_dir}")
    print(f"PDB input dir: {input_pdb_dir}")
    print(f"FFT bridge input dir: {input_fft_data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Motif cluster assignments file: {motif_cluster_assignments_file}")
    print(f"DSSP executable requested: {dssp_executable_requested}")
    print(f"DSSP executable used: {dssp_executable_used if dssp_executable_used else 'None'}")

    motif_df = load_motif_cluster_assignments(motif_cluster_assignments_file)
    validate_motif_cluster_assignments(motif_df, motif_cluster_assignments_file.name)

    fft_bridge_file_map = discover_fft_bridge_files(input_fft_data_dir)

    proteins = sorted(motif_df["protein"].astype(str).unique().tolist())
    print(f"Proteins encountered: {len(proteins)}")

    window_validation_frames: list[pd.DataFrame] = []
    skipped_windows_frames: list[pd.DataFrame] = []
    skipped_proteins_frames: list[pd.DataFrame] = []

    for protein in proteins:
        protein_windows_df = motif_df.loc[motif_df["protein"].astype(str) == protein].copy()
        protein_window_validation_df, protein_skipped_windows_df, protein_skipped_proteins_df = (
            process_protein_windows(
                protein=protein,
                protein_windows_df=protein_windows_df,
                input_pdb_dir=input_pdb_dir,
                fft_bridge_file_map=fft_bridge_file_map,
                dssp_executable=dssp_executable_used,
                verbose=args.verbose,
            )
        )
        window_validation_frames.append(protein_window_validation_df)
        skipped_windows_frames.append(protein_skipped_windows_df)
        skipped_proteins_frames.append(protein_skipped_proteins_df)

    window_validation_df = pd.concat(window_validation_frames, ignore_index=True)
    skipped_windows_df = pd.concat(skipped_windows_frames, ignore_index=True)
    skipped_proteins_df = pd.concat(skipped_proteins_frames, ignore_index=True)

    skipped_windows_df = skipped_windows_df.reindex(columns=SKIPPED_WINDOWS_COLUMNS)
    skipped_proteins_df = skipped_proteins_df.reindex(columns=SKIPPED_PROTEINS_COLUMNS)

    window_validation_df = window_validation_df.sort_values(
        by=["protein", "chain", "window_start_seq_index", "window_label"],
        kind="mergesort",
    ).reset_index(drop=True)

    skipped_windows_df = skipped_windows_df.sort_values(
        by=["protein", "chain", "window_label"],
        kind="mergesort",
    ).reset_index(drop=True)

    skipped_proteins_df = skipped_proteins_df.sort_values(
        by=["protein"],
        kind="mergesort",
    ).reset_index(drop=True)

    confusion_df = build_confusion_matrix(window_validation_df)
    cluster_label_mapping_df = build_cluster_label_mapping(window_validation_df, confusion_df)
    cluster_purity_df = build_cluster_purity(window_validation_df, cluster_label_mapping_df)
    cluster_purity_df["dssp_label"] = pd.Categorical(
        cluster_purity_df["dssp_label"],
        categories=VALID_DSSP_LABELS,
        ordered=True,
    )
    cluster_purity_df = cluster_purity_df.sort_values(
        by=["motif_cluster_id", "dssp_label"],
        kind="mergesort",
    ).reset_index(drop=True)
    cluster_purity_df["dssp_label"] = cluster_purity_df["dssp_label"].astype(str)

    cluster_label_mapping_df = cluster_label_mapping_df.sort_values(
        by=["motif_cluster_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    confusion_df = confusion_df.sort_values(by=["motif_cluster_id"], kind="mergesort").reset_index(drop=True)
    precision_recall_df = build_precision_recall(window_validation_df, cluster_label_mapping_df)

    n_input_windows = int(len(motif_df))
    n_validated_windows = int((window_validation_df["validation_status"] == STATUS_VALIDATED).sum())
    n_metric_eligible_windows = int(window_validation_df["validation_eligible_for_metrics"].sum())
    n_scored_windows = (
        int(precision_recall_df["n_scored_windows"].iloc[0])
        if not precision_recall_df.empty
        else 0
    )
    n_skipped_windows = int(len(skipped_windows_df))
    n_skipped_proteins = int(len(skipped_proteins_df))
    n_successful_proteins = int(len(proteins) - n_skipped_proteins)

    metadata = {
        "module_name": "Theta_FFT Module 4 Validation",
        "script_name": script_path.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_comparison_dir": str(input_comparison_dir),
        "input_pdb_dir": str(input_pdb_dir),
        "input_fft_data_dir": str(input_fft_data_dir),
        "output_dir": str(output_dir),
        "motif_cluster_assignments_file": str(motif_cluster_assignments_file),
        "dssp_executable_requested": dssp_executable_requested,
        "dssp_executable_used": dssp_executable_used,
        "n_input_windows": n_input_windows,
        "n_validated_windows": n_validated_windows,
        "n_metric_eligible_windows": n_metric_eligible_windows,
        "n_scored_windows": n_scored_windows,
        "n_skipped_windows": n_skipped_windows,
        "n_skipped_proteins": n_skipped_proteins,
        "motif_clusters_seen": sorted(window_validation_df["motif_cluster_id"].astype(str).unique().tolist()),
        "simplified_dssp_mapping": {
            "H": ["H", "G", "I", "P"],
            "E": ["E", "B"],
            "L": "All other DSSP codes, including blank/space, dash, T, S, C, and any non-helix/non-sheet symbol.",
        },
        "notes": [
            "input = precomputed motif_cluster_assignments.csv",
            "no FFT was recomputed in this module",
            "no motif clustering was recomputed in this module",
            "windows were mapped to DSSP using fft_data seq_index-to-residue bridges",
            "DSSP labels were simplified to H/E/L using H,G,I,P -> H and E,B -> E",
            "majority vote was performed at the window level",
            "no plots were generated in this module",
            "window_start_seq_index and window_end_seq_index were reconstructed from window_center_seq_index and window_length when absent in the input comparison table",
        ],
    }

    write_outputs(
        output_dir=output_dir,
        window_validation_df=window_validation_df,
        cluster_purity_df=cluster_purity_df,
        cluster_label_mapping_df=cluster_label_mapping_df,
        precision_recall_df=precision_recall_df,
        confusion_df=confusion_df,
        skipped_proteins_df=skipped_proteins_df,
        skipped_windows_df=skipped_windows_df,
        metadata=metadata,
    )

    print(f"Proteins DSSP-validated successfully: {n_successful_proteins}")
    print(f"Proteins skipped: {n_skipped_proteins}")
    print(f"Input windows loaded: {n_input_windows}")
    print(f"Windows with VALIDATED status: {n_validated_windows}")
    print(f"Metric-eligible windows: {n_metric_eligible_windows}")
    print(f"Precision/recall scored windows: {n_scored_windows}")
    print(f"Main output file: {output_dir / 'window_validation.csv'}")
    print(f"Cluster purity file: {output_dir / 'cluster_purity.csv'}")
    print(f"Cluster mapping file: {output_dir / 'cluster_label_mapping.csv'}")
    print(f"Precision/recall file: {output_dir / 'precision_recall_by_class.csv'}")
    print(f"Confusion matrix file: {output_dir / 'confusion_matrix.csv'}")
    print(f"Skipped proteins file: {output_dir / 'skipped_proteins.csv'}")
    print(f"Skipped windows file: {output_dir / 'skipped_windows.csv'}")
    print(f"Metadata file: {output_dir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
