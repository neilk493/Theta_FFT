import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "protein",
    "chain",
    "seq_index",
    "res_i",
    "res_j",
    "theta_signed",
    "has_gap_before",
]

MIN_SEGMENT_LENGTH = 8


def validate_columns(df, source_path):
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        print(
            f"WARNING: Skipping {source_path.name} because required columns are missing: "
            f"{', '.join(missing)}"
        )
        return False
    return True


def split_into_segments(chain_df):
    chain_sorted = chain_df.sort_values("seq_index").reset_index(drop=True)
    segment_starts = [0]
    gap_rows = chain_sorted.index[chain_sorted["has_gap_before"].astype(int) == 1].tolist()
    for row_index in gap_rows:
        if row_index != 0:
            segment_starts.append(row_index)

    segments = []
    for i, start_row in enumerate(segment_starts):
        end_row = segment_starts[i + 1] if i + 1 < len(segment_starts) else len(chain_sorted)
        segment_df = chain_sorted.iloc[start_row:end_row].reset_index(drop=True)
        if not segment_df.empty:
            segments.append(segment_df)
    return segments


def compute_segment_spectrum(segment_df):
    signal = segment_df["theta_signed"].to_numpy(dtype=np.float64)
    n_residues = signal.shape[0]
    signal_rad = np.deg2rad(signal)
    signal_unwrapped_rad = np.unwrap(signal_rad)
    signal = np.rad2deg(signal_unwrapped_rad)
    indices = np.arange(n_residues, dtype=np.float64)
    slope, intercept = np.polyfit(indices, signal, 1)
    trend = slope * indices + intercept
    signal = signal - trend
    window = np.hanning(n_residues)
    windowed = signal * window
    fft_result = np.fft.rfft(windowed)
    power = np.abs(fft_result) ** 2
    freqs = np.fft.rfftfreq(n_residues, d=1.0)
    amplitude = np.abs(fft_result)
    non_dc_mask = freqs > 0
    total_power_non_dc = float(np.sum(power[non_dc_mask]))
    helix_mask = (freqs >= 0.20) & (freqs <= 0.35)
    sheet_mask = (freqs >= 0.45) & (freqs <= 0.50)
    if total_power_non_dc > 0:
        helix_band_fraction = float(np.sum(power[helix_mask]) / total_power_non_dc)
        sheet_band_fraction = float(np.sum(power[sheet_mask]) / total_power_non_dc)
    else:
        helix_band_fraction = 0.0
        sheet_band_fraction = 0.0
    dominant_index = int(np.argmax(power[1:]) + 1)
    dominant_freq = float(freqs[dominant_index])

    return {
        "freqs": freqs,
        "power": power,
        "amplitude": amplitude,
        "dominant_freq": dominant_freq,
        "helix_band_fraction": helix_band_fraction,
        "sheet_band_fraction": sheet_band_fraction,
    }


def segment_to_rows(protein, chain, segment_id, segment_df, spectrum):
    seg_start = int(segment_df["seq_index"].iloc[0])
    seg_end = int(segment_df["seq_index"].iloc[-1])
    seg_length = int(len(segment_df))
    dominant_freq = spectrum["dominant_freq"]
    helix_band_fraction = spectrum["helix_band_fraction"]
    sheet_band_fraction = spectrum["sheet_band_fraction"]

    rows = []
    for freq, power, amplitude in zip(
        spectrum["freqs"], spectrum["power"], spectrum["amplitude"]
    ):
        rows.append(
            {
                "protein": protein,
                "chain": chain,
                "segment_id": segment_id,
                "seg_start_seq_index": seg_start,
                "seg_end_seq_index": seg_end,
                "seg_length": seg_length,
                "dominant_freq_cycles_per_residue": dominant_freq,
                "helix_band_power_fraction": helix_band_fraction,
                "sheet_band_power_fraction": sheet_band_fraction,
                "freq_cycles_per_residue": float(freq),
                "power": float(np.log10(power + 1e-6)),
                "amplitude": float(amplitude),
            }
        )
    return rows


def save_protein_plot(protein, segment_plots, output_png_path):
    n_segments = len(segment_plots)
    fig, axes = plt.subplots(
        nrows=n_segments,
        ncols=1,
        figsize=(10, 3 * n_segments),
        squeeze=False,
    )

    for ax, plot_data in zip(axes[:, 0], segment_plots):
        log_power = np.log10(plot_data["power"] + 1e-6)
        ax.plot(
            plot_data["freqs"],
            log_power,
            color="tab:blue",
            linewidth=1.5,
        )
        dominant_idx = int(np.argmin(np.abs(plot_data["freqs"] - plot_data["dominant_freq"])))
        ax.plot(
            plot_data["freqs"][dominant_idx],
            log_power[dominant_idx],
            "ko",
            markersize=6,
            label=f"dominant: {plot_data['dominant_freq']:.4f} cyc/res",
        )
        ax.set_title(
            f"{protein} chain {plot_data['chain']} segment {plot_data['segment_id']} "
            f"(residues {plot_data['seg_start']}–{plot_data['seg_end']}, N={plot_data['seg_length']})"
        )
        ax.set_xlabel("Frequency (cycles/residue)")
        ax.set_ylabel("Log10 Power")
        ax.legend()

    plt.tight_layout()
    fig.savefig(output_png_path, dpi=150)
    plt.close(fig)


def process_file(input_csv_path, output_dir):
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as exc:
        print(f"WARNING: Skipping {input_csv_path.name} because it could not be read: {exc}")
        return

    if not validate_columns(df, input_csv_path):
        return

    protein_values = df["protein"].dropna().astype(str).unique()
    protein = protein_values[0] if len(protein_values) > 0 else input_csv_path.name.replace("_fft_data.csv", "")

    output_rows = []
    segment_plots = []
    valid_segments = 0

    for chain, chain_df in df.groupby("chain", sort=True):
        segments = split_into_segments(chain_df)
        segment_id = 0
        for segment_df in segments:
            segment_length = len(segment_df)
            seg_start = int(segment_df["seq_index"].iloc[0])
            if segment_length < MIN_SEGMENT_LENGTH:
                print(
                    f"WARNING: Skipping segment in {protein} chain {chain} starting at seq_index "
                    f"{seg_start} — only {segment_length} residues (minimum is 8)."
                )
                continue

            spectrum = compute_segment_spectrum(segment_df)
            output_rows.extend(
                segment_to_rows(
                    protein=protein,
                    chain=chain,
                    segment_id=segment_id,
                    segment_df=segment_df,
                    spectrum=spectrum,
                )
            )
            segment_plots.append(
                {
                    "chain": chain,
                    "segment_id": segment_id,
                    "seg_start": seg_start,
                    "seg_end": int(segment_df["seq_index"].iloc[-1]),
                    "seg_length": segment_length,
                    "freqs": spectrum["freqs"],
                    "power": spectrum["power"],
                    "dominant_freq": spectrum["dominant_freq"],
                }
            )
            valid_segments += 1
            segment_id += 1

    if valid_segments == 0:
        print(f"WARNING: No valid segments found for {protein}; no output files written.")
        return

    output_df = pd.DataFrame(
        output_rows,
        columns=[
            "protein",
            "chain",
            "segment_id",
            "seg_start_seq_index",
            "seg_end_seq_index",
            "seg_length",
            "dominant_freq_cycles_per_residue",
            "helix_band_power_fraction",
            "sheet_band_power_fraction",
            "freq_cycles_per_residue",
            "power",
            "amplitude",
        ],
    )

    output_csv_path = output_dir / f"{protein}_global_spectrum.csv"
    output_png_path = output_dir / f"{protein}_global_spectrum.png"
    output_df.to_csv(output_csv_path, index=False)
    save_protein_plot(protein, segment_plots, output_png_path)
    print(f"Processed {protein}: {valid_segments} segments written to {output_csv_path}")


def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    script_dir = Path(__file__).parent.resolve()
    fft_data_dir = script_dir / "fft_data"
    output_dir = script_dir / "output" / "global_spectra"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(fft_data_dir.glob("*_fft_data.csv"))
    if not input_files:
        print(f"ERROR: No _fft_data.csv files found in {fft_data_dir}")
        sys.exit(1)

    for input_csv_path in input_files:
        process_file(input_csv_path, output_dir)


if __name__ == "__main__":
    main()
