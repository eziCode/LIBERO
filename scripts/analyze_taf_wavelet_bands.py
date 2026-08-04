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


CHANNELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
REGIMES = ("low_load", "steady_load", "loading", "unloading", "spatial_change")
EVENTS = ("loading", "unloading", "spatial_change")


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
    parser.add_argument("--output", type=Path, default=Path("results/taf_force_analysis/raw"))
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument(
        "--center-force",
        action="store_true",
        help="Subtract each episode/channel median before decomposition.",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--event-window-seconds",
        type=float,
        default=2.0,
        help="Seconds retained before and after each pressure-defined event onset.",
    )
    parser.add_argument(
        "--event-refractory-seconds",
        type=float,
        default=1.0,
        help="Minimum separation between event onsets of the same type.",
    )
    parser.add_argument(
        "--baseline-gap-seconds",
        type=float,
        default=0.5,
        help="Gap between the end of the local pre-event baseline and event onset.",
    )
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


def swt_energy(signal: np.ndarray, wavelet: str, levels: int) -> tuple[np.ndarray, list[str]]:
    multiple = 2**levels
    if len(signal) % multiple:
        raise ValueError(
            f"SWT input length {len(signal)} is not divisible by {multiple}; "
            "truncate the episode before calling swt_energy"
        )
    # trim_approx returns [A_L, D_L, ..., D_1]. norm=True makes scale-energy
    # comparisons less sensitive to redundant SWT coefficient magnitudes.
    coeffs = pywt.swt(signal, wavelet, level=levels, trim_approx=True, norm=True)
    coeffs = [np.asarray(c) for c in coeffs]
    energy = np.stack([c * c for c in coeffs], axis=1)
    names = [f"A{levels}"] + [f"D{i}" for i in range(levels, 0, -1)]
    return energy, names


def bootstrap_band_contrasts(
    episode_profiles: pd.DataFrame, iterations: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for keys, group in episode_profiles.groupby(["regime", "channel", "band"], sort=False):
        values = group["standardized_difference"].to_numpy()
        draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
        regime, channel, band = keys
        rows.append(
            {
                "regime": regime,
                "channel": channel,
                "band": band,
                "episodes": len(values),
                "mean_standardized_difference": float(values.mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def event_onsets(
    labels: np.ndarray, event: str, radius: int, refractory: int
) -> np.ndarray:
    mask = labels == event
    candidates = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    candidates = candidates[(candidates >= radius) & (candidates < len(labels) - radius)]
    kept: list[int] = []
    for candidate in candidates:
        if not kept or candidate - kept[-1] >= refractory:
            kept.append(int(candidate))
    return np.asarray(kept, dtype=int)


def bootstrap_event_series(
    episode_series: pd.DataFrame, iterations: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for keys, group in episode_series.groupby(["event", "channel", "band"], sort=False):
        table = group.pivot(index="episode", columns="relative_sample", values="value").sort_index(axis=1)
        values = table.to_numpy()
        if not len(values):
            continue
        sample_indices = rng.integers(0, len(values), size=(iterations, len(values)))
        draws = values[sample_indices].mean(axis=1)
        mean = values.mean(axis=0)
        low, high = np.quantile(draws, [0.025, 0.975], axis=0)
        event, channel, band = keys
        event_count = int(group.groupby("episode")["event_count"].first().sum())
        for column, mean_value, low_value, high_value in zip(table.columns, mean, low, high):
            rows.append(
                {
                    "event": event,
                    "channel": channel,
                    "band": band,
                    "relative_sample": int(column),
                    "episodes": len(values),
                    "events": event_count,
                    "mean": float(mean_value),
                    "ci_low": float(low_value),
                    "ci_high": float(high_value),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_post_window(
    episode_series: pd.DataFrame, fps: float, iterations: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    post = episode_series[
        (episode_series["relative_sample"] >= 0)
        & (episode_series["relative_sample"] <= round(0.5 * fps))
    ]
    episode_means = (
        post.groupby(["episode", "event", "channel", "band"], as_index=False)["value"]
        .mean()
    )
    rows: list[dict[str, object]] = []
    for keys, group in episode_means.groupby(["event", "channel", "band"], sort=False):
        values = group["value"].to_numpy()
        draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
        event, channel, band = keys
        rows.append(
            {
                "event": event,
                "channel": channel,
                "band": band,
                "episodes": len(values),
                "mean_post_0_5_db": float(values.mean()),
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


def plot_band_contrast_heatmaps(summary: pd.DataFrame, output: Path, bands: list[str]) -> None:
    for channel in CHANNELS:
        table = (
            summary[summary["channel"] == channel]
            .pivot(index="regime", columns="band", values="mean_standardized_difference")
            .reindex(index=REGIMES, columns=bands)
        )
        fig, ax = plt.subplots(figsize=(7, 4.2))
        limit = max(1.0, float(np.nanmax(np.abs(table.to_numpy()))))
        image = ax.imshow(table.to_numpy(), aspect="auto", vmin=-limit, vmax=limit, cmap="coolwarm")
        ax.set_xticks(range(len(bands)), bands)
        ax.set_yticks(range(len(REGIMES)), REGIMES)
        for y in range(len(REGIMES)):
            for x in range(len(bands)):
                value = table.iloc[y, x]
                if np.isfinite(value):
                    ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
        ax.set_title(f"{channel}: regime energy minus rest-of-sequence energy")
        fig.colorbar(image, ax=ax, label="Standardized difference (episode SD)")
        fig.tight_layout()
        fig.savefig(output / f"band_contrast_heatmap_{channel}.png", dpi=160)
        plt.close(fig)


def plot_event_series(summary: pd.DataFrame, output: Path, fps: float, bands: list[str]) -> None:
    colors = dict(zip(bands, ("#555555", "#3b4cc0", "#2fb47c", "#fdae61", "#d7191c")))
    for channel in CHANNELS:
        fig, axes = plt.subplots(len(EVENTS), 1, figsize=(9, 10), sharex=True)
        for row, event in enumerate(EVENTS):
            subset = summary[(summary["channel"] == channel) & (summary["event"] == event)]
            for band in bands:
                curve = subset[subset["band"] == band].sort_values("relative_sample")
                x = curve["relative_sample"].to_numpy() / fps
                axes[row].plot(x, curve["mean"], label=band, color=colors[band], lw=1.4)
                axes[row].fill_between(x, curve["ci_low"], curve["ci_high"], color=colors[band], alpha=0.12)
            axes[row].axvline(0, color="#cc3333", ls="--", lw=1)
            axes[row].axhline(0, color="black", ls=":", lw=1)
            axes[row].grid(alpha=0.2)
            axes[row].set_ylabel(f"{event}\nenergy change (dB)")
        axes[0].legend(ncol=len(bands), fontsize=8)
        axes[-1].set_xlabel("Seconds from pressure-defined event onset")
        fig.suptitle(f"{channel}: unsmoothed event-centered band energy\n(local pre-event mean log energy = 0 dB)")
        fig.tight_layout()
        fig.savefig(output / f"event_centered_{channel}.png", dpi=160)
        plt.close(fig)


def plot_average_scalograms(
    summary: pd.DataFrame, output: Path, fps: float, bands: list[str]
) -> None:
    scalogram_dir = output / "event_scalograms"
    scalogram_dir.mkdir(parents=True, exist_ok=True)
    for event in EVENTS:
        event_data = summary[summary["event"] == event]
        values = []
        for channel in CHANNELS:
            table = (
                event_data[event_data["channel"] == channel]
                .pivot(index="band", columns="relative_sample", values="mean")
                .reindex(index=bands)
            )
            values.append(table.to_numpy())
        limit = max(1.0, float(np.nanpercentile(np.abs(np.stack(values)), 98)))
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
        image = None
        for ax, channel, matrix in zip(axes.flat, CHANNELS, values):
            samples = np.sort(event_data["relative_sample"].unique())
            extent = [samples[0] / fps, samples[-1] / fps, len(bands) - 0.5, -0.5]
            image = ax.imshow(
                matrix, aspect="auto", extent=extent, cmap="coolwarm", vmin=-limit, vmax=limit
            )
            ax.axvline(0, color="black", ls="--", lw=1)
            ax.set_title(channel)
            ax.set_yticks(range(len(bands)), bands)
            ax.set_xlabel("Seconds from onset")
        axes[0, 0].set_ylabel("Wavelet component")
        axes[1, 0].set_ylabel("Wavelet component")
        fig.colorbar(image, ax=axes, label="Energy relative to local pre-event baseline (dB)")
        fig.suptitle(f"{event}: event-triggered average scalogram")
        fig.subplots_adjust(left=0.07, right=0.90, bottom=0.09, top=0.90, wspace=0.12, hspace=0.25)
        fig.savefig(scalogram_dir / f"average_{event}.png", dpi=180)
        plt.close(fig)


def plot_representative_scalograms(
    windows: dict[tuple[int, str, int, str], np.ndarray],
    metrics: pd.DataFrame,
    output: Path,
    fps: float,
    bands: list[str],
) -> None:
    scalogram_dir = output / "event_scalograms"
    for event in EVENTS:
        scores = (
            metrics[(metrics["event"] == event) & metrics["band"].isin(["D2", "D1"])]
            .groupby(["episode", "onset_frame"])["post_0_5_mean_db"]
            .mean()
            .sort_values()
        )
        if scores.empty:
            continue
        episode, onset = scores.index[len(scores) // 2]
        matrices = [windows[(int(episode), event, int(onset), channel)] for channel in CHANNELS]
        limit = max(1.0, float(np.nanpercentile(np.abs(np.stack(matrices)), 98)))
        radius = (matrices[0].shape[0] - 1) // 2
        extent = [-radius / fps, radius / fps, len(bands) - 0.5, -0.5]
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
        image = None
        for ax, channel, matrix in zip(axes.flat, CHANNELS, matrices):
            image = ax.imshow(
                matrix.T, aspect="auto", extent=extent, cmap="coolwarm", vmin=-limit, vmax=limit
            )
            ax.axvline(0, color="black", ls="--", lw=1)
            ax.set_title(channel)
            ax.set_yticks(range(len(bands)), bands)
            ax.set_xlabel("Seconds from onset")
        axes[0, 0].set_ylabel("Wavelet component")
        axes[1, 0].set_ylabel("Wavelet component")
        fig.colorbar(image, ax=axes, label="Energy relative to local pre-event baseline (dB)")
        fig.suptitle(f"Representative {event} event: episode {episode}, frame {onset}")
        fig.subplots_adjust(left=0.07, right=0.90, bottom=0.09, top=0.90, wspace=0.12, hspace=0.25)
        fig.savefig(scalogram_dir / f"representative_{event}.png", dpi=180)
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
    band_contrast_rows: list[dict[str, object]] = []
    event_series_rows: list[dict[str, object]] = []
    event_metric_rows: list[dict[str, object]] = []
    individual_event_windows: dict[tuple[int, str, int, str], np.ndarray] = {}
    event_count_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    threshold_output: dict[str, dict[str, float]] = {}
    truncation_output: dict[str, int] = {}
    bands: list[str] | None = None
    event_radius = max(1, round(args.event_window_seconds * fps))
    event_refractory = max(1, round(args.event_refractory_seconds * fps))
    baseline_gap = max(1, round(args.baseline_gap_seconds * fps))
    if baseline_gap >= event_radius:
        raise ValueError("baseline-gap-seconds must be smaller than event-window-seconds")

    for episode, episode_frame in frame.groupby("episode_index", sort=True):
        episode_frame = episode_frame.sort_values("frame_index")
        original_length = len(episode_frame)
        usable_length = original_length - (original_length % (2**args.levels))
        if usable_length == 0:
            raise ValueError(
                f"Episode {episode} has only {original_length} frames, fewer than required "
                f"for {args.levels} SWT levels"
            )
        episode_frame = episode_frame.iloc[:usable_length]
        truncation_output[str(episode)] = original_length - usable_length
        pressure = pressure_array(episode_frame["observation.pressure_matrix"])
        wrench = vector_array(episode_frame["observation.force_torque"])
        time = episode_frame["timestamp"].to_numpy(dtype=float)
        labels, thresholds, pressure_signals = derive_pressure_regimes(pressure)
        threshold_output[str(episode)] = thresholds
        onsets = {
            event: event_onsets(labels, event, event_radius, event_refractory)
            for event in EVENTS
        }
        for event, indices in onsets.items():
            event_count_rows.append(
                {"episode": int(episode), "event": event, "events": int(len(indices))}
            )

        channel_energies: list[np.ndarray] = []
        for channel_index, channel in enumerate(CHANNELS):
            signal = wrench[:, channel_index].copy()
            if args.center_force:
                signal -= np.median(signal)
            energy, current_bands = swt_energy(signal, args.wavelet, args.levels)
            bands = current_bands
            channel_energies.append(energy)
            episode_band_std = np.maximum(energy.std(axis=0), 1e-12)
            for regime in REGIMES:
                mask = labels == regime
                if not mask.any():
                    continue
                regime_mean = energy[mask].mean(axis=0)
                rest_mean = energy[~mask].mean(axis=0)
                standardized_difference = (regime_mean - rest_mean) / episode_band_std
                for band_index, band in enumerate(bands):
                    band_contrast_rows.append(
                        {
                            "episode": int(episode),
                            "regime": regime,
                            "channel": channel,
                            "band": band,
                            "regime_mean_energy": float(regime_mean[band_index]),
                            "rest_mean_energy": float(rest_mean[band_index]),
                            "energy_difference": float(regime_mean[band_index] - rest_mean[band_index]),
                            "standardized_difference": float(standardized_difference[band_index]),
                        }
                    )

            relative_samples = np.arange(-event_radius, event_radius + 1)
            for event, indices in onsets.items():
                if not len(indices):
                    continue
                standardized_windows = []
                for event_number, index in enumerate(indices):
                    window = energy[index - event_radius : index + event_radius + 1]
                    energy_floor = np.maximum(energy.mean(axis=0) * 1e-12, 1e-30)
                    log_window = 10.0 * np.log10(window + energy_floor[None, :])
                    baseline_energy = energy[index - event_radius : index - baseline_gap]
                    log_baseline = 10.0 * np.log10(baseline_energy + energy_floor[None, :])
                    baseline_log_mean = log_baseline.mean(axis=0)
                    baseline_centered_window = log_window - baseline_log_mean[None, :]
                    standardized_windows.append(baseline_centered_window)
                    individual_event_windows[(int(episode), event, int(index), channel)] = baseline_centered_window
                    post_half = (relative_samples >= 0) & (relative_samples <= round(0.5 * fps))
                    post_one = (relative_samples >= 0) & (relative_samples <= round(1.0 * fps))
                    for band_index, band in enumerate(bands):
                        post_values = baseline_centered_window[post_one, band_index]
                        peak_offset = int(np.argmax(post_values))
                        event_metric_rows.append(
                            {
                                "episode": int(episode),
                                "event": event,
                                "event_number": event_number,
                                "onset_frame": int(index),
                                "channel": channel,
                                "band": band,
                                "baseline_mean_energy": float(baseline_energy[:, band_index].mean()),
                                "baseline_mean_log_energy_db": float(baseline_log_mean[band_index]),
                                "onset_db": float(baseline_centered_window[event_radius, band_index]),
                                "post_0_5_mean_db": float(baseline_centered_window[post_half, band_index].mean()),
                                "post_1_mean_db": float(post_values.mean()),
                                "post_1_peak_db": float(post_values[peak_offset]),
                                "post_1_peak_seconds": float(relative_samples[post_one][peak_offset] / fps),
                            }
                        )
                event_windows = np.stack(standardized_windows).mean(axis=0)
                for band_index, band in enumerate(bands):
                    for relative_sample, value in zip(relative_samples, event_windows[:, band_index]):
                        event_series_rows.append(
                            {
                                "episode": int(episode), "event": event, "event_count": len(indices),
                                "channel": channel, "band": band,
                                "relative_sample": int(relative_sample), "value": float(value),
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
    band_contrasts = pd.DataFrame(band_contrast_rows)
    contrast_summary = bootstrap_band_contrasts(band_contrasts, args.bootstrap, args.seed)
    event_episode_series = pd.DataFrame(event_series_rows)
    event_metrics = pd.DataFrame(event_metric_rows)
    event_summary = bootstrap_event_series(event_episode_series, args.bootstrap, args.seed + 2)
    post_window_summary = bootstrap_post_window(
        event_episode_series, fps, args.bootstrap, args.seed + 3
    )
    band_contrasts.to_csv(args.output / "band_contrasts_by_episode.csv", index=False)
    contrast_summary.to_csv(args.output / "band_contrast_summary.csv", index=False)
    event_episode_series.to_csv(args.output / "event_centered_episode_series.csv", index=False)
    event_summary.to_csv(args.output / "event_centered_summary.csv", index=False)
    post_window_summary.to_csv(args.output / "event_post_0_5s_summary.csv", index=False)
    event_metrics.to_csv(args.output / "event_level_metrics.csv", index=False)
    pd.DataFrame(event_count_rows).to_csv(args.output / "event_counts.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(args.output / "regime_counts.csv", index=False)
    (args.output / "pressure_thresholds.json").write_text(json.dumps(threshold_output, indent=2))
    plot_band_contrast_heatmaps(contrast_summary, args.output, bands)
    plot_event_series(event_summary, args.output, fps, bands)
    plot_average_scalograms(event_summary, args.output, fps, bands)
    plot_representative_scalograms(
        individual_event_windows, event_metrics, args.output, fps, bands
    )

    analyzed_frames = int(
        sum(
            len(group) - (len(group) % (2**args.levels))
            for _, group in frame.groupby("episode_index")
        )
    )
    metadata = {
        "data": str(args.data),
        "episodes": int(frame["episode_index"].nunique()),
        "frames": analyzed_frames,
        "original_frames": int(len(frame)),
        "fps": fps,
        "wavelet": args.wavelet,
        "levels": args.levels,
        "bands_hz_approx": {
            f"D{level}": [fps / (2 ** (level + 1)), fps / (2**level)]
            for level in range(1, args.levels + 1)
        }
        | {f"A{args.levels}": [0.0, fps / (2 ** (args.levels + 1))]},
        "energy_definition": "Unsmoothed squared SWT coefficient C_j[t]^2.",
        "regime_contrast_definition": (
            "For each band, (mean energy inside regime - mean energy outside regime) divided by the "
            "episode-wide standard deviation of that band's energy. No energy ratio is used."
        ),
        "event_series_definition": (
            "For each event and band, unsmoothed C_j[t]^2 was converted to log energy (10 log10) "
            "and centered by its local mean log energy from -event_window_seconds to "
            "-baseline_gap_seconds. Events were then "
            "averaged within episode before episode-level bootstrap aggregation."
        ),
        "event_window_seconds": args.event_window_seconds,
        "baseline_gap_seconds": args.baseline_gap_seconds,
        "event_refractory_seconds": args.event_refractory_seconds,
        "event_definition": "Onsets of pressure-derived loading, unloading, and spatial-change regimes.",
        "center_force": args.center_force,
        "truncation": {
            "policy": "Drop trailing real samples until episode length is divisible by 2**levels.",
            "dropped_frames_by_episode": truncation_output,
            "analyzed_frames": analyzed_frames,
        },
        "regime_definition": "Per-episode pressure-map quantiles; no wrench values used.",
    }
    (args.output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Analyzed {metadata['frames']} frames from {metadata['episodes']} episodes at {fps:.1f} Hz")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
