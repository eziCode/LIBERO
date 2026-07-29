#!/usr/bin/env python3
"""Explore whether pressure-defined contact regimes occupy distinct wavelet bands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/libero-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy.ndimage import uniform_filter1d


CHANNELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
REGIMES = ("low_load", "steady_load", "loading", "unloading", "spatial_change")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive contact regimes from TaF pressure maps and measure the "
            "wavelet-band energy of the independently recorded 6D wrench."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "datasets/taf_subset/taf_dataset/gs_mini/gs_mini_obj1/"
            "data/chunk-000/file-000.parquet"
        ),
        help="LeRobot parquet containing pressure_matrix and force_torque.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/taf_wavelet_analysis"))
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument(
        "--energy-window-seconds",
        type=float,
        default=0.5,
        help="Width of the moving average applied to squared coefficients.",
    )
    parser.add_argument(
        "--center-force",
        action="store_true",
        help="Subtract each episode/channel median before decomposition.",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def pressure_array(series: pd.Series) -> np.ndarray:
    # Arrow returns each 12x12 matrix as an object array of 12 numeric rows.
    return np.stack([np.vstack(value).astype(np.float64) for value in series])


def vector_array(series: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(value, dtype=np.float64) for value in series])


def derive_pressure_regimes(pressure: np.ndarray) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    total = pressure.sum(axis=(1, 2))
    total_delta = np.gradient(total)

    yy, xx = np.mgrid[: pressure.shape[1], : pressure.shape[2]]
    safe_total = np.maximum(total, 1e-9)
    centroid_x = (pressure * xx).sum(axis=(1, 2)) / safe_total
    centroid_y = (pressure * yy).sum(axis=(1, 2)) / safe_total
    centroid_speed = np.hypot(np.gradient(centroid_x), np.gradient(centroid_y))
    map_change = np.r_[
        0.0,
        np.linalg.norm(np.diff(pressure, axis=0).reshape(len(pressure) - 1, -1), axis=1),
    ]

    thresholds = {
        "low_load_q20": float(np.quantile(total, 0.20)),
        "loading_delta_q80": float(np.quantile(total_delta, 0.80)),
        "unloading_delta_q20": float(np.quantile(total_delta, 0.20)),
        "steady_abs_delta_q30": float(np.quantile(np.abs(total_delta), 0.30)),
        "map_change_q90": float(np.quantile(map_change, 0.90)),
        "centroid_speed_q90": float(np.quantile(centroid_speed, 0.90)),
    }

    labels = np.full(len(total), "other", dtype=object)
    labels[total <= thresholds["low_load_q20"]] = "low_load"
    labels[
        (total > thresholds["low_load_q20"])
        & (np.abs(total_delta) <= thresholds["steady_abs_delta_q30"])
    ] = "steady_load"
    labels[total_delta >= thresholds["loading_delta_q80"]] = "loading"
    labels[total_delta <= thresholds["unloading_delta_q20"]] = "unloading"
    labels[
        (map_change >= thresholds["map_change_q90"])
        | (centroid_speed >= thresholds["centroid_speed_q90"])
    ] = "spatial_change"

    signals = {
        "total_pressure": total,
        "pressure_delta": total_delta,
        "map_change": map_change,
        "centroid_speed": centroid_speed,
    }
    return labels, thresholds, signals


def swt_energy(
    signal: np.ndarray, wavelet: str, levels: int, smooth_samples: int
) -> tuple[np.ndarray, list[str]]:
    multiple = 2**levels
    pad = (-len(signal)) % multiple
    padded = np.pad(signal, (0, pad), mode="wrap") if pad else signal
    # trim_approx returns [A_L, D_L, ..., D_1]. norm=True makes scale-energy
    # comparisons less sensitive to redundant SWT coefficient magnitudes.
    coeffs = pywt.swt(padded, wavelet, level=levels, trim_approx=True, norm=True)
    coeffs = [np.asarray(c[: len(signal)]) for c in coeffs]
    energy = np.stack(
        [uniform_filter1d(c * c, size=smooth_samples, mode="nearest") for c in coeffs],
        axis=1,
    )
    names = [f"A{levels}"] + [f"D{i}" for i in range(levels, 0, -1)]
    return energy, names


def bootstrap_profiles(
    episode_profiles: pd.DataFrame, iterations: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    group_cols = ["regime", "channel", "band"]
    for keys, group in episode_profiles.groupby(group_cols, sort=False):
        values = group["energy_fraction"].to_numpy()
        if not len(values):
            continue
        draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "episodes": len(values),
                "mean_energy_fraction": float(values.mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def plot_episode(
    output: Path,
    episode: int,
    time: np.ndarray,
    wrench: np.ndarray,
    labels: np.ndarray,
    pressure_signals: dict[str, np.ndarray],
    fz_energy: np.ndarray,
    bands: list[str],
) -> None:
    colors = {
        "low_load": "#c7dcef",
        "steady_load": "#b8e0c5",
        "loading": "#ffd59a",
        "unloading": "#dfc2f2",
        "spatial_change": "#ffaaa5",
    }
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(time, pressure_signals["total_pressure"], color="black", lw=0.8)
    axes[0].set_ylabel("Total pressure")
    axes[1].plot(time, wrench[:, 2], color="#3366aa", lw=0.8)
    axes[1].set_ylabel("Fz (N)")
    axes[2].plot(time, pressure_signals["map_change"], color="#7a3e9d", lw=0.8)
    axes[2].set_ylabel("Map change")
    for band_index, band in enumerate(bands):
        axes[3].plot(time, fz_energy[:, band_index], lw=0.8, label=band)
    axes[3].set_yscale("symlog", linthresh=1e-7)
    axes[3].set_ylabel("Fz band energy")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(ncol=len(bands), fontsize=8)

    for ax in axes:
        for regime, color in colors.items():
            mask = labels == regime
            starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
            ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
            for start, end in zip(starts, ends):
                ax.axvspan(time[start], time[end], color=color, alpha=0.12, lw=0)
        ax.grid(alpha=0.2)
    fig.suptitle(f"Episode {episode}: pressure-defined regimes and Fz wavelet energy")
    fig.tight_layout()
    fig.savefig(output / f"episode_{episode}_overview.png", dpi=160)
    plt.close(fig)


def plot_heatmaps(summary: pd.DataFrame, output: Path, bands: list[str]) -> None:
    for channel in CHANNELS:
        table = (
            summary[summary["channel"] == channel]
            .pivot(index="regime", columns="band", values="mean_energy_fraction")
            .reindex(index=REGIMES, columns=bands)
        )
        fig, ax = plt.subplots(figsize=(7, 4.2))
        image = ax.imshow(table.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="magma")
        ax.set_xticks(range(len(bands)), bands)
        ax.set_yticks(range(len(REGIMES)), REGIMES)
        for y in range(len(REGIMES)):
            for x in range(len(bands)):
                value = table.iloc[y, x]
                if np.isfinite(value):
                    ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
        ax.set_title(f"{channel}: normalized wavelet energy by pressure regime")
        fig.colorbar(image, ax=ax, label="Energy fraction")
        fig.tight_layout()
        fig.savefig(output / f"regime_band_heatmap_{channel}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.data)
    required = {"observation.pressure_matrix", "observation.force_torque", "episode_index", "timestamp"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    fps = float(1.0 / np.median(np.diff(np.sort(frame["timestamp"].unique()))))
    smooth_samples = max(1, round(args.energy_window_seconds * fps))
    profile_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    threshold_output: dict[str, dict[str, float]] = {}
    bands: list[str] | None = None

    for episode, episode_frame in frame.groupby("episode_index", sort=True):
        episode_frame = episode_frame.sort_values("frame_index")
        pressure = pressure_array(episode_frame["observation.pressure_matrix"])
        wrench = vector_array(episode_frame["observation.force_torque"])
        time = episode_frame["timestamp"].to_numpy(dtype=float)
        labels, thresholds, pressure_signals = derive_pressure_regimes(pressure)
        threshold_output[str(episode)] = thresholds

        channel_energies: list[np.ndarray] = []
        for channel_index, channel in enumerate(CHANNELS):
            signal = wrench[:, channel_index].copy()
            if args.center_force:
                signal -= np.median(signal)
            energy, current_bands = swt_energy(signal, args.wavelet, args.levels, smooth_samples)
            bands = current_bands
            channel_energies.append(energy)
            denominator = np.maximum(energy.sum(axis=1, keepdims=True), 1e-12)
            fractions = energy / denominator
            for regime in REGIMES:
                mask = labels == regime
                if not mask.any():
                    continue
                mean_profile = fractions[mask].mean(axis=0)
                for band, value in zip(bands, mean_profile):
                    profile_rows.append(
                        {
                            "episode": int(episode),
                            "regime": regime,
                            "channel": channel,
                            "band": band,
                            "energy_fraction": float(value),
                        }
                    )

        for regime in (*REGIMES, "other"):
            count = int(np.sum(labels == regime))
            regime_rows.append(
                {"episode": int(episode), "regime": regime, "frames": count, "fraction": count / len(labels)}
            )

        if int(episode) == int(frame["episode_index"].min()):
            plot_episode(
                args.output,
                int(episode),
                time,
                wrench,
                labels,
                pressure_signals,
                channel_energies[2],
                bands,
            )

    assert bands is not None
    episode_profiles = pd.DataFrame(profile_rows)
    summary = bootstrap_profiles(episode_profiles, args.bootstrap, args.seed)
    episode_profiles.to_csv(args.output / "episode_band_profiles.csv", index=False)
    summary.to_csv(args.output / "regime_band_summary.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(args.output / "regime_counts.csv", index=False)
    (args.output / "pressure_thresholds.json").write_text(json.dumps(threshold_output, indent=2))
    plot_heatmaps(summary, args.output, bands)

    metadata = {
        "data": str(args.data),
        "episodes": int(frame["episode_index"].nunique()),
        "frames": int(len(frame)),
        "fps": fps,
        "wavelet": args.wavelet,
        "levels": args.levels,
        "bands_hz_approx": {
            f"D{level}": [fps / (2 ** (level + 1)), fps / (2**level)]
            for level in range(1, args.levels + 1)
        }
        | {f"A{args.levels}": [0.0, fps / (2 ** (args.levels + 1))]},
        "energy_window_seconds": args.energy_window_seconds,
        "center_force": args.center_force,
        "regime_definition": "Per-episode pressure-map quantiles; no wrench values used.",
    }
    (args.output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Analyzed {metadata['frames']} frames from {metadata['episodes']} episodes at {fps:.1f} Hz")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
