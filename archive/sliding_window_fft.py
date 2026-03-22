import sys
import warnings
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WINDOW_SIZE = 55
STEP_SIZE = 1
MIN_SEGMENT_LENGTH = WINDOW_SIZE

REQUIRED_COLUMNS = [
    "protein",
    "chain",
    "seq_index",
    "res_i",
    "res_j",
    "theta_signed",
    "has_gap_before",
]

FEATURE_COLUMNS = [
    "protein",
    "chain",
    "segment_id",
    "seg_start_seq_index",
    "seg_end_seq_index",
    "seg_length",
    "window_idx",
    "window_center_seq_index",
    "window_start_seq_index",
    "window_end_seq_index",
    "theta_arithmetic_mean",
    "theta_variance",
    "theta_circular_mean",
    "theta_R",
    "dominant_freq_cycles_per_residue",
    "total_power",
    "spectral_entropy",
    "helix_band_power_fraction",
    "sheet_band_power_fraction",
    "inter_band_power_fraction",
    "helix_band_peak_freq",
    "sheet_band_peak_freq",
    "helix_band_weighted_mean_freq",
    "sheet_band_weighted_mean_freq",
    "autocorr_lag_1",
    "autocorr_lag_2",
    "autocorr_lag_3",
    "autocorr_lag_4",
    "autocorr_lag_5",
    "autocorr_lag_6",
    "autocorr_lag_7",
    "autocorr_lag_8",
    "autocorr_lag_9",
    "autocorr_lag_10",
]


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
    for idx, start_row in enumerate(segment_starts):
        end_row = segment_starts[idx + 1] if idx + 1 < len(segment_starts) else len(chain_sorted)
        segment_df = chain_sorted.iloc[start_row:end_row].reset_index(drop=True)
        if not segment_df.empty:
            segments.append(segment_df)
    return segments


def compute_frequency_axis():
    return np.fft.rfftfreq(WINDOW_SIZE, d=1.0)


def compute_theta_statistics(raw_window):
    theta_arithmetic_mean = float(np.mean(raw_window))
    theta_variance = float(np.var(raw_window))

    sin_mean = np.mean(np.sin(np.deg2rad(raw_window)))
    cos_mean = np.mean(np.cos(np.deg2rad(raw_window)))
    theta_circular_mean = float(np.degrees(np.arctan2(sin_mean, cos_mean)))
    theta_r = float(np.sqrt(sin_mean**2 + cos_mean**2))

    return {
        "theta_arithmetic_mean": theta_arithmetic_mean,
        "theta_variance": theta_variance,
        "theta_circular_mean": theta_circular_mean,
        "theta_R": theta_r,
    }


def compute_window_spectrum(signal, freqs):
    n = len(signal)
    indices = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(indices, signal, 1)
    trend = slope * indices + intercept
    signal_detrended = signal - trend

    window_func = np.hanning(WINDOW_SIZE)
    windowed = signal_detrended * window_func
    fft_result = np.fft.rfft(windowed)
    power = np.abs(fft_result) ** 2

    dominant_freq = float(freqs[np.argmax(power[1:]) + 1])
    total_power = float(np.sum(power[1:]))
    non_dc_power = power[1:]
    if total_power > 0:
        p_normalized = non_dc_power / total_power
        spectral_entropy = float(-np.sum(p_normalized * np.log2(p_normalized + 1e-12)))
    else:
        spectral_entropy = 0.0

    helix_mask = (freqs >= 0.20) & (freqs <= 0.35)
    sheet_mask = (freqs >= 0.45) & (freqs <= 0.50)
    inter_mask = (freqs > 0.35) & (freqs < 0.45)

    helix_band_power = float(np.sum(power[helix_mask]))
    sheet_band_power = float(np.sum(power[sheet_mask]))

    if total_power > 0:
        helix_band_power_fraction = helix_band_power / total_power
        sheet_band_power_fraction = float(np.sum(power[sheet_mask]) / total_power)
        inter_band_power_fraction = float(np.sum(power[inter_mask]) / total_power)
    else:
        helix_band_power_fraction = 0.0
        sheet_band_power_fraction = 0.0
        inter_band_power_fraction = 0.0

    if np.any(helix_mask) and helix_band_power > 0:
        helix_band_peak_freq = float(freqs[helix_mask][np.argmax(power[helix_mask])])
    else:
        helix_band_peak_freq = float("nan")

    if np.any(sheet_mask) and sheet_band_power > 0:
        sheet_band_peak_freq = float(freqs[sheet_mask][np.argmax(power[sheet_mask])])
    else:
        sheet_band_peak_freq = float("nan")

    if helix_band_power > 0:
        helix_band_weighted_mean_freq = float(
            np.sum(freqs[helix_mask] * power[helix_mask]) / helix_band_power
        )
    else:
        helix_band_weighted_mean_freq = float("nan")

    if sheet_band_power > 0:
        sheet_band_weighted_mean_freq = float(
            np.sum(freqs[sheet_mask] * power[sheet_mask]) / sheet_band_power
        )
    else:
        sheet_band_weighted_mean_freq = float("nan")

    autocorr_full = np.correlate(signal_detrended, signal_detrended, mode="full")
    mid = len(autocorr_full) // 2
    autocorr_normalized = autocorr_full[mid:] / (autocorr_full[mid] + 1e-12)
    autocorr_lags = {
        f"autocorr_lag_{k}": float(autocorr_normalized[k]) for k in range(1, 11)
    }

    return {
        "power": power.astype(np.float64, copy=False),
        "dominant_freq": dominant_freq,
        "total_power": total_power,
        "spectral_entropy": spectral_entropy,
        "helix_band_power_fraction": helix_band_power_fraction,
        "sheet_band_power_fraction": sheet_band_power_fraction,
        "inter_band_power_fraction": inter_band_power_fraction,
        "helix_band_peak_freq": helix_band_peak_freq,
        "sheet_band_peak_freq": sheet_band_peak_freq,
        "helix_band_weighted_mean_freq": helix_band_weighted_mean_freq,
        "sheet_band_weighted_mean_freq": sheet_band_weighted_mean_freq,
        "autocorr_lags": autocorr_lags,
    }


def analyze_segment(protein, chain, segment_id, segment_df, freqs):
    seg_start = int(segment_df["seq_index"].iloc[0])
    seg_end = int(segment_df["seq_index"].iloc[-1])
    seg_length = int(len(segment_df))

    if seg_length < MIN_SEGMENT_LENGTH:
        print(
            f"WARNING: Skipping segment in {protein} chain {chain} starting at seq_index "
            f"{seg_start} — only {seg_length} residues, need at least {WINDOW_SIZE} for one window."
        )
        return [], None, None

    raw_theta = segment_df["theta_signed"].to_numpy(dtype=np.float64)
    theta_rad = np.deg2rad(raw_theta)
    theta_unwrapped_rad = np.unwrap(theta_rad)
    theta_unwrapped_deg = np.rad2deg(theta_unwrapped_rad)

    n_windows = (seg_length - WINDOW_SIZE) // STEP_SIZE + 1
    spectrogram = np.zeros((n_windows, WINDOW_SIZE // 2 + 1), dtype=np.float64)
    window_centers = np.zeros(n_windows, dtype=int)
    feature_rows = []

    for window_idx, start_idx in enumerate(range(0, seg_length - WINDOW_SIZE + 1, STEP_SIZE)):
        window_df = segment_df.iloc[start_idx : start_idx + WINDOW_SIZE]
        raw_window = raw_theta[start_idx : start_idx + WINDOW_SIZE]
        unwrapped_window = theta_unwrapped_deg[start_idx : start_idx + WINDOW_SIZE]
        theta_stats = compute_theta_statistics(raw_window)
        spectrum = compute_window_spectrum(unwrapped_window, freqs)

        center_seq_index = int(
            segment_df["seq_index"].iloc[start_idx + WINDOW_SIZE // 2]
        )
        window_start_seq_index = int(window_df["seq_index"].iloc[0])
        window_end_seq_index = int(window_df["seq_index"].iloc[-1])

        spectrogram[window_idx, :] = spectrum["power"]
        window_centers[window_idx] = center_seq_index

        feature_rows.append(
            {
                "protein": protein,
                "chain": chain,
                "segment_id": segment_id,
                "seg_start_seq_index": seg_start,
                "seg_end_seq_index": seg_end,
                "seg_length": seg_length,
                "window_idx": window_idx,
                "window_center_seq_index": center_seq_index,
                "window_start_seq_index": window_start_seq_index,
                "window_end_seq_index": window_end_seq_index,
                "theta_arithmetic_mean": theta_stats["theta_arithmetic_mean"],
                "theta_variance": theta_stats["theta_variance"],
                "theta_circular_mean": theta_stats["theta_circular_mean"],
                "theta_R": theta_stats["theta_R"],
                "dominant_freq_cycles_per_residue": spectrum["dominant_freq"],
                "total_power": spectrum["total_power"],
                "spectral_entropy": spectrum["spectral_entropy"],
                "helix_band_power_fraction": spectrum["helix_band_power_fraction"],
                "sheet_band_power_fraction": spectrum["sheet_band_power_fraction"],
                "inter_band_power_fraction": spectrum["inter_band_power_fraction"],
                "helix_band_peak_freq": spectrum["helix_band_peak_freq"],
                "sheet_band_peak_freq": spectrum["sheet_band_peak_freq"],
                "helix_band_weighted_mean_freq": spectrum["helix_band_weighted_mean_freq"],
                "sheet_band_weighted_mean_freq": spectrum["sheet_band_weighted_mean_freq"],
                **spectrum["autocorr_lags"],
            }
        )

    return feature_rows, spectrogram, window_centers


def save_spectrogram_plot(protein, plot_segments, freqs, output_png_path):
    n_total_segments = len(plot_segments)
    fig, axes = plt.subplots(
        nrows=n_total_segments,
        ncols=1,
        figsize=(12, 4 * n_total_segments),
        squeeze=False,
    )

    for ax, plot_data in zip(axes[:, 0], plot_segments):
        spectrogram = plot_data["spectrogram"]
        window_centers = plot_data["window_centers"]
        log_spectrogram = np.log10(spectrogram.T + 1e-6)

        im = ax.imshow(
            log_spectrogram,
            origin="lower",
            aspect="auto",
            cmap="plasma",
            extent=[window_centers[0], window_centers[-1], freqs[0], freqs[-1]],
        )
        ax.set_xlabel("Residue position (seq_index)")
        ax.set_ylabel("Frequency (cycles/residue)")
        ax.set_title(
            f"{protein} chain {plot_data['chain']} segment {plot_data['segment_id']} | "
            f"N={plot_data['seg_length']} residues | window={WINDOW_SIZE} step={STEP_SIZE}"
        )
        ax.axhline(
            y=0.500,
            color="cyan",
            linestyle="--",
            linewidth=1.2,
            label="β-sheet (0.500)",
        )
        ax.axhline(
            y=0.278,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label="α-helix (0.278)",
        )
        ax.axhline(
            y=0.333,
            color="orange",
            linestyle="--",
            linewidth=1.0,
            label="3₁₀-helix (0.333)",
        )
        ax.axhline(
            y=0.227,
            color="pink",
            linestyle="--",
            linewidth=1.0,
            label="π-helix (0.227)",
        )
        ax.legend(loc="upper right", fontsize=8)
        plt.colorbar(im, ax=ax, label="Log10 Power")

    plt.tight_layout()
    fig.savefig(output_png_path, dpi=150)
    plt.close(fig)


def process_file(input_csv_path, output_dir, freqs):
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as exc:
        print(f"WARNING: Skipping {input_csv_path.name} because it could not be read: {exc}")
        return

    if not validate_columns(df, input_csv_path):
        return

    protein_values = df["protein"].dropna().astype(str).unique()
    protein = (
        protein_values[0]
        if len(protein_values) > 0
        else input_csv_path.name.replace("_fft_data.csv", "")
    )

    feature_rows = []
    plot_segments = []
    arrays_dict = {"freqs": freqs.astype(np.float64, copy=False)}
    total_windows = 0
    total_segments = 0

    for chain, chain_df in df.groupby("chain", sort=True):
        segments = split_into_segments(chain_df)
        segment_id = 0
        for segment_df in segments:
            segment_rows, spectrogram, window_centers = analyze_segment(
                protein=protein,
                chain=chain,
                segment_id=segment_id,
                segment_df=segment_df,
                freqs=freqs,
            )
            if spectrogram is None:
                continue

            feature_rows.extend(segment_rows)
            plot_segments.append(
                {
                    "chain": chain,
                    "segment_id": segment_id,
                    "seg_length": len(segment_df),
                    "spectrogram": spectrogram,
                    "window_centers": window_centers,
                }
            )
            arrays_dict[f"segment_{chain}_{segment_id}"] = spectrogram
            arrays_dict[f"window_centers_{chain}_{segment_id}"] = window_centers
            total_windows += spectrogram.shape[0]
            total_segments += 1
            segment_id += 1

    if total_segments == 0:
        print(f"WARNING: No valid segments found for {protein}; no output files written.")
        return

    output_df = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    output_csv_path = output_dir / f"{protein}_window_features.csv"
    output_npz_path = output_dir / f"{protein}_spectrogram.npz"
    output_png_path = output_dir / f"{protein}_spectrogram.png"

    output_df.to_csv(output_csv_path, index=False)
    np.savez(output_npz_path, **arrays_dict)
    save_spectrogram_plot(protein, plot_segments, freqs, output_png_path)

    if protein == "1TEN":
        mean_sheet_fraction = float(output_df["sheet_band_power_fraction"].mean())
        has_sheet_signal = bool((output_df["sheet_band_power_fraction"] > 0).any())
        spectrogram_sheet_mask = (freqs >= 0.45) & (freqs <= 0.50)
        has_sheet_band_power = any(
            np.any(segment_data["spectrogram"][:, spectrogram_sheet_mask] > 0)
            for segment_data in plot_segments
        )
        if not has_sheet_signal and not has_sheet_band_power:
            print(
                "DIAGNOSTIC: 1TEN shows no detectable sheet signal after unwrapping and "
                f"WINDOW_SIZE={WINDOW_SIZE}. Mean sheet_band_power_fraction = {mean_sheet_fraction:.6f}"
            )

    print(
        f"Processed {protein}: {total_windows} windows across {total_segments} segments "
        f"— written to {output_csv_path}"
    )


def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    fft_data_dir = SCRIPT_DIR / "fft_data"
    output_dir = SCRIPT_DIR / "output" / "spectrograms"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(fft_data_dir.glob("*_fft_data.csv"))
    if not input_files:
        print(f"ERROR: No _fft_data.csv files found in {fft_data_dir}")
        sys.exit(1)

    freqs = compute_frequency_axis()

    for input_csv_path in input_files:
        process_file(input_csv_path, output_dir, freqs)


if __name__ == "__main__":
    main()
