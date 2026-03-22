"""Reformat an existing 3D t-SNE coordinate table for use in Desmos 3D.

This script exports grouped coordinate files by protein and by motif cluster,
writing both archival CSV files and headerless paste-ready TXT files. It only
reorganizes an already-generated 3D t-SNE table and does not recompute any
analysis, clustering, dimensionality reduction, or biological interpretation.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"tsne_1", "tsne_2", "tsne_3", "protein", "motif_cluster_id"}
OPTIONAL_SORT_COLUMNS = [
    "protein",
    "chain",
    "segment_label",
    "window_center_seq_index",
    "window_label",
]
PROTEIN_EXPORT_ORDER = ["2HHB", "1GZM", "1TEN", "1FNA", "2IGF", "1PKK", "1UBQ", "1LYZ", "2PTN"]
TAB10_HEX = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]
DESMOS_EXPRESSION_HINT = "(x_1,y_1,z_1)"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Desmos export workflow."""

    parser = argparse.ArgumentParser(
        description=(
            "Reformat an existing 3D t-SNE coordinate table into Desmos-ready "
            "exports split by protein and motif cluster."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Theta_FFT/output/comparison_plots/tsne_3d_coordinates.csv"),
        help="Primary 3D t-SNE coordinate CSV path.",
    )
    parser.add_argument(
        "--fallback-input",
        type=Path,
        default=Path("Theta_FFT/output/comparison/tsne_3d_coordinates.csv"),
        help="Fallback 3D t-SNE coordinate CSV path if the primary file is missing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Theta_FFT/output/desmos_3d_exports"),
        help="Root directory for Desmos-ready exports.",
    )
    parser.add_argument(
        "--max-points-per-file",
        type=int,
        default=950,
        help="Maximum number of points per exported CSV/TXT file.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal precision for exported coordinates.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-group export progress.",
    )
    return parser.parse_args()


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Create the required output directory structure."""

    output_dir.mkdir(parents=True, exist_ok=True)
    directories = {
        "root": output_dir,
        "by_protein": output_dir / "by_protein",
        "by_motif_cluster": output_dir / "by_motif_cluster",
        "manifests": output_dir / "manifests",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def resolve_cli_path(path_value: Path, script_dir: Path) -> Path:
    """Resolve a CLI path robustly relative to the script location."""

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


def resolve_input_path(primary_input: Path, fallback_input: Path | None) -> Path:
    """Resolve the preferred input path, falling back when needed."""

    if primary_input.exists():
        return primary_input
    if fallback_input is not None and fallback_input.exists():
        return fallback_input

    message = f"3D t-SNE coordinate table not found. Tried: {primary_input}"
    if fallback_input is not None:
        message += f" and fallback: {fallback_input}"
    raise FileNotFoundError(message)


def load_coordinates(input_path: Path) -> pd.DataFrame:
    """Load the existing coordinate table from disk."""

    return pd.read_csv(input_path)


def _normalize_motif_cluster_id(value: object) -> str:
    """Normalize motif cluster labels for stable naming and sorting."""

    text = str(value).strip()
    motif_match = re.fullmatch(r"motif(\d+)", text, flags=re.IGNORECASE)
    if motif_match:
        return f"motif{int(motif_match.group(1)):03d}"

    numeric_match = re.fullmatch(r"[+-]?\d+(?:\.0+)?", text)
    if numeric_match:
        return f"motif{int(float(text)):03d}"

    return text


def validate_and_clean(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Validate required schema, coerce coordinates, and drop unusable rows."""

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if df.empty:
        raise ValueError("Input coordinate table is empty")

    cleaned = df.copy()

    for column in ["tsne_1", "tsne_2", "tsne_3"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "window_center_seq_index" in cleaned.columns:
        cleaned["window_center_seq_index"] = pd.to_numeric(
            cleaned["window_center_seq_index"], errors="coerce"
        )

    cleaned["protein"] = cleaned["protein"].astype("string").str.strip()
    cleaned["motif_cluster_id"] = cleaned["motif_cluster_id"].astype("string").str.strip()

    missing_group_mask = (
        cleaned["protein"].isna()
        | cleaned["motif_cluster_id"].isna()
        | (cleaned["protein"] == "")
        | (cleaned["motif_cluster_id"] == "")
    )
    dropped_missing_groups = int(missing_group_mask.sum())
    if dropped_missing_groups:
        print(
            "Dropped "
            f"{dropped_missing_groups} rows with missing protein or motif_cluster_id values"
        )
        cleaned = cleaned.loc[~missing_group_mask].copy()

    missing_coordinate_mask = cleaned[["tsne_1", "tsne_2", "tsne_3"]].isna().any(axis=1)
    dropped_missing_coordinates = int(missing_coordinate_mask.sum())
    if dropped_missing_coordinates:
        print(
            "Dropped "
            f"{dropped_missing_coordinates} rows with missing or non-numeric coordinate values"
        )
        cleaned = cleaned.loc[~missing_coordinate_mask].copy()

    if cleaned.empty:
        raise ValueError("No rows remained after validation and cleaning")

    cleaned["protein"] = cleaned["protein"].astype(str)
    cleaned["motif_cluster_id"] = cleaned["motif_cluster_id"].map(_normalize_motif_cluster_id)

    if verbose:
        print(f"Validated coordinate table with {len(cleaned)} retained rows")

    return cleaned


def sort_group_df(group_df: pd.DataFrame) -> pd.DataFrame:
    """Stably sort a group using the preferred metadata columns when available."""

    sort_columns = [column for column in OPTIONAL_SORT_COLUMNS if column in group_df.columns]
    if not sort_columns:
        return group_df.reset_index(drop=True)
    return group_df.sort_values(by=sort_columns, kind="mergesort").reset_index(drop=True)


def hex_to_rgb_tuple(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color code like #RRGGBB into an RGB tuple."""

    normalized = hex_color.strip().lstrip("#")
    if len(normalized) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got: {hex_color}")
    return tuple(int(normalized[index : index + 2], 16) for index in (0, 2, 4))


def get_protein_color_map() -> dict[str, str]:
    """Return the fixed protein palette requested for Desmos exports."""

    return {
        "2HHB": "#C0392B",
        "1GZM": "#E74C3C",
        "1TEN": "#2980B9",
        "1FNA": "#5DADE2",
        "2IGF": "#1A5276",
        "1PKK": "#148F77",
        "1UBQ": "#7D6608",
        "1LYZ": "#82E0AA",
        "2PTN": "#7F8C8D",
    }


def _cluster_sort_key(cluster_id: str) -> tuple[int, int, str]:
    """Sort motif cluster labels numerically when they follow motifNNN naming."""

    text = str(cluster_id)
    motif_match = re.fullmatch(r"motif(\d+)", text, flags=re.IGNORECASE)
    if motif_match:
        return (0, int(motif_match.group(1)), text)

    numeric_match = re.fullmatch(r"[+-]?\d+(?:\.0+)?", text)
    if numeric_match:
        return (1, int(float(text)), text)

    return (2, np.iinfo(np.int64).max, text)


def get_motif_color_map(cluster_ids: list[str]) -> dict[str, str]:
    """Return a stable motif color mapping based on matplotlib tab10 hex colors."""

    sorted_cluster_ids = sorted({str(cluster_id) for cluster_id in cluster_ids}, key=_cluster_sort_key)
    return {
        cluster_id: TAB10_HEX[index % len(TAB10_HEX)]
        for index, cluster_id in enumerate(sorted_cluster_ids)
    }


def chunk_dataframe(df: pd.DataFrame, max_points: int) -> list[pd.DataFrame]:
    """Split a dataframe into ordered chunks with at most max_points rows."""

    if max_points <= 0:
        raise ValueError("max_points must be greater than 0")
    if df.empty:
        return []
    return [df.iloc[start : start + max_points].reset_index(drop=True) for start in range(0, len(df), max_points)]


def _build_export_xyz(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 3-column coordinate export dataframe."""

    return pd.DataFrame(
        {
            "x": df["tsne_1"].to_numpy(dtype=float),
            "y": df["tsne_2"].to_numpy(dtype=float),
            "z": df["tsne_3"].to_numpy(dtype=float),
        }
    )


def write_archival_csv(df: pd.DataFrame, output_path: Path, precision: int) -> Path:
    """Write the archival CSV file with header x,y,z."""

    export_df = _build_export_xyz(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(
        output_path,
        index=False,
        columns=["x", "y", "z"],
        float_format=f"%.{precision}f",
        lineterminator="\n",
    )
    return output_path


def write_desmos_paste_txt(df: pd.DataFrame, output_path: Path, precision: int) -> Path:
    """Write the headerless tab-delimited Desmos paste file."""

    export_df = _build_export_xyz(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(
        output_path,
        index=False,
        header=False,
        sep="\t",
        columns=["x", "y", "z"],
        float_format=f"%.{precision}f",
        lineterminator="\n",
    )
    return output_path


def _ordered_groups_for_export(
    df: pd.DataFrame, group_col: str, export_type: str, color_map: dict[str, str]
) -> list[str]:
    """Return a deterministic export order for grouped outputs."""

    existing_groups = {str(value) for value in df[group_col].astype(str).unique().tolist()}

    if export_type == "by_protein":
        ordered = [group for group in PROTEIN_EXPORT_ORDER if group in existing_groups]
        extras = sorted(existing_groups - set(PROTEIN_EXPORT_ORDER))
        return ordered + extras

    motif_order = sorted(existing_groups, key=_cluster_sort_key)
    known_order = [group for group in color_map if group in existing_groups]
    if known_order:
        extras = [group for group in motif_order if group not in set(known_order)]
        return known_order + extras
    return motif_order


def export_grouped_files(
    df: pd.DataFrame,
    group_col: str,
    output_subdir: Path,
    color_map: dict[str, str],
    export_type: str,
    max_points: int,
    precision: int,
    verbose: bool = False,
) -> pd.DataFrame:
    """Export grouped CSV/TXT files and return a manifest dataframe."""

    manifest_rows: list[dict[str, object]] = []
    output_root = output_subdir.parent
    ordered_groups = _ordered_groups_for_export(df, group_col, export_type, color_map)

    for group_name in ordered_groups:
        group_df = df.loc[df[group_col].astype(str) == str(group_name)].copy()
        if group_df.empty:
            continue

        sorted_group = sort_group_df(group_df)
        chunks = chunk_dataframe(sorted_group, max_points=max_points)
        if verbose:
            print(f"Exporting {export_type} group {group_name}: {len(sorted_group)} rows across {len(chunks)} file(s)")

        hex_color = color_map.get(str(group_name), "#808080")
        rgb_tuple = hex_to_rgb_tuple(hex_color)
        rgb_expression = f"rgb({rgb_tuple[0]},{rgb_tuple[1]},{rgb_tuple[2]})"

        for index, chunk_df in enumerate(chunks, start=1):
            part_label = f"part{index:02d}"
            use_part_suffix = len(chunks) > 1
            if use_part_suffix:
                csv_name = f"{group_name}_desmos_{part_label}.csv"
                txt_name = f"{group_name}_desmos_{part_label}_paste.txt"
            else:
                csv_name = f"{group_name}_desmos.csv"
                txt_name = f"{group_name}_desmos_paste.txt"

            csv_path = write_archival_csv(chunk_df, output_subdir / csv_name, precision=precision)
            txt_path = write_desmos_paste_txt(chunk_df, output_subdir / txt_name, precision=precision)

            manifest_rows.append(
                {
                    "export_type": export_type,
                    "group_name": str(group_name),
                    "part": part_label,
                    "n_points": int(len(chunk_df)),
                    "csv_path": csv_path.relative_to(output_root).as_posix(),
                    "paste_txt_path": txt_path.relative_to(output_root).as_posix(),
                    "suggested_hex_color": hex_color,
                    "suggested_rgb_expression": rgb_expression,
                    "desmos_expression_hint": DESMOS_EXPRESSION_HINT,
                }
            )

    columns = [
        "export_type",
        "group_name",
        "part",
        "n_points",
        "csv_path",
        "paste_txt_path",
        "suggested_hex_color",
        "suggested_rgb_expression",
        "desmos_expression_hint",
    ]
    return pd.DataFrame(manifest_rows, columns=columns)


def write_manifest(manifest_df: pd.DataFrame, output_path: Path) -> Path:
    """Write a manifest CSV to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_path, index=False, lineterminator="\n")
    return output_path


def _format_color_lines(mapping: dict[str, str], ordered_keys: Iterable[str]) -> list[str]:
    """Format human-readable color guide lines."""

    lines: list[str] = []
    for key in ordered_keys:
        if key not in mapping:
            continue
        rgb = hex_to_rgb_tuple(mapping[key])
        lines.append(f"{key}: {mapping[key]} | rgb({rgb[0]},{rgb[1]},{rgb[2]})")
    return lines


def write_instruction_files(
    manifests_dir: Path, protein_colors: dict[str, str], motif_colors: dict[str, str]
) -> list[Path]:
    """Write Desmos import instructions, color guide, and expression templates."""

    import_instructions_path = manifests_dir / "DESMOS_3D_IMPORT_INSTRUCTIONS.txt"
    color_guide_path = manifests_dir / "DESMOS_3D_COLOR_GUIDE.txt"
    expression_templates_path = manifests_dir / "desmos_expression_templates.txt"

    import_instructions_text = "\n".join(
        [
            "Desmos 3D import workflow",
            "",
            "1. Open Desmos 3D and create or open a blank graph.",
            "2. Open one *_desmos_paste.txt file and paste its contents directly into a Desmos table.",
            "3. If the table does not auto-plot, add a new expression using the table variable names in point form, for example:",
            "   (x_1,y_1,z_1)",
            "4. Repeat for each protein group or motif-cluster group that you want to inspect.",
            "5. Set the point color manually in Desmos using the suggested colors listed in DESMOS_3D_COLOR_GUIDE.txt.",
            "6. For best readability, import either the protein-split export set or the motif-split export set, not both at once.",
            "7. If a group was split into multiple parts, import all parts for that group.",
            "",
            "Note: actual table variable names may vary depending on paste order, so x_1/y_1/z_1 is only a generic example.",
        ]
    )
    import_instructions_path.write_text(import_instructions_text + "\n", encoding="utf-8", newline="\n")

    protein_lines = _format_color_lines(protein_colors, PROTEIN_EXPORT_ORDER)
    motif_lines = _format_color_lines(motif_colors, sorted(motif_colors, key=_cluster_sort_key))
    color_guide_text = "\n".join(
        [
            "Desmos 3D color guide",
            "",
            "Protein colors",
            *protein_lines,
            "",
            "Motif cluster colors",
            *motif_lines,
            "",
            "Colors are applied manually in Desmos. The CSV and TXT exports contain only numeric coordinates.",
        ]
    )
    color_guide_path.write_text(color_guide_text + "\n", encoding="utf-8", newline="\n")

    expression_templates_text = "\n".join(
        [
            "Desmos expression templates",
            "",
            "Protein import examples",
            "(x_1,y_1,z_1)",
            "(x_2,y_2,z_2)",
            "",
            "Motif import examples",
            "(x_3,y_3,z_3)",
            "(x_4,y_4,z_4)",
            "",
            "Generic examples",
            "(x_1,y_1,z_1)",
            "(x_2,y_2,z_2)",
            "(x_3,y_3,z_3)",
            "",
            "Actual table numbering depends on the order in which tables are pasted into Desmos.",
        ]
    )
    expression_templates_path.write_text(
        expression_templates_text + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return [import_instructions_path, color_guide_path, expression_templates_path]


def main() -> None:
    """Run the export-only Desmos formatting workflow."""

    args = parse_args()
    if args.max_points_per_file < 50:
        raise ValueError("--max-points-per-file must be at least 50")
    if not 3 <= args.precision <= 10:
        raise ValueError("--precision must be between 3 and 10 inclusive")

    script_dir = Path(__file__).resolve().parent
    primary_input = resolve_cli_path(args.input, script_dir)
    fallback_input = resolve_cli_path(args.fallback_input, script_dir) if args.fallback_input else None
    output_dir = resolve_cli_path(args.output_dir, script_dir)
    output_paths = ensure_output_dirs(output_dir)

    input_path = resolve_input_path(primary_input, fallback_input)
    print(f"Using input file: {input_path}")

    df = load_coordinates(input_path)
    cleaned_df = validate_and_clean(df, verbose=args.verbose)
    print(f"Retained rows: {len(cleaned_df)}")

    protein_colors = get_protein_color_map()
    motif_ids = sorted(cleaned_df["motif_cluster_id"].astype(str).unique().tolist(), key=_cluster_sort_key)
    motif_colors = get_motif_color_map(motif_ids)

    protein_manifest = export_grouped_files(
        df=cleaned_df,
        group_col="protein",
        output_subdir=output_paths["by_protein"],
        color_map=protein_colors,
        export_type="by_protein",
        max_points=args.max_points_per_file,
        precision=args.precision,
        verbose=args.verbose,
    )
    motif_manifest = export_grouped_files(
        df=cleaned_df,
        group_col="motif_cluster_id",
        output_subdir=output_paths["by_motif_cluster"],
        color_map=motif_colors,
        export_type="by_motif_cluster",
        max_points=args.max_points_per_file,
        precision=args.precision,
        verbose=args.verbose,
    )

    protein_manifest_path = write_manifest(
        protein_manifest,
        output_paths["manifests"] / "desmos_export_manifest_by_protein.csv",
    )
    motif_manifest_path = write_manifest(
        motif_manifest,
        output_paths["manifests"] / "desmos_export_manifest_by_motif_cluster.csv",
    )
    instruction_paths = write_instruction_files(
        manifests_dir=output_paths["manifests"],
        protein_colors=protein_colors,
        motif_colors=motif_colors,
    )

    print(
        "Protein export files written: "
        f"{len(protein_manifest) * 2} files across {len(protein_manifest)} part(s)"
    )
    print(
        "Motif export files written: "
        f"{len(motif_manifest) * 2} files across {len(motif_manifest)} part(s)"
    )
    print(f"Protein manifest: {protein_manifest_path}")
    print(f"Motif manifest: {motif_manifest_path}")
    for path in instruction_paths:
        print(f"Instruction/helper file: {path}")


if __name__ == "__main__":
    main()
