"""
Pipeline F: Place field and place cell analysis.

Performs:
1. Movement and spatial filtering
2. Transient detection (C or S mode)
3. Place field calculation with Gaussian smoothing
4. Place cell identification via mutual information shuffle test
5. Direction-specific analysis (L/R)

Output: Place field plots and results MAT file.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from ..config import get_output_path
from ..utils.io_utils import load_mat_file, save_mat_file


def calculate_speed(position: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Calculate instantaneous speed from position and timestamps."""
    if len(position) < 2:
        return np.zeros_like(position)

    dp = np.diff(position)
    dt = np.diff(timestamps)
    dt[dt == 0] = 1e-6

    speed = np.abs(dp / dt)
    speed = np.append(speed, speed[-1])

    return speed


def detect_transients_C_mode(
    activity: np.ndarray,
    sn: float,
    frame_rate: float,
    temporal_window_ms: float,
    amplitude_threshold_sigma: float,
) -> np.ndarray:
    """Detect Ca2+ transients for C mode using peak detection."""
    distance_frames = int(temporal_window_ms / 1000 * frame_rate)
    threshold = amplitude_threshold_sigma * sn

    peaks, _ = find_peaks(activity, height=threshold, distance=distance_frames)
    return peaks


def detect_transients_S_mode(activity: np.ndarray) -> np.ndarray:
    """Detect transients for S mode (deconvolved spikes)."""
    MIN_SPIKE_MAGNITUDE = 1e-6
    return np.where(activity > MIN_SPIKE_MAGNITUDE)[0]


def apply_movement_and_spatial_filter(
    position: np.ndarray,
    speed: np.ndarray,
    min_speed: float,
    reward_zone: float,
    track_length: float,
) -> np.ndarray:
    """Apply movement speed filter and exclude reward zones."""
    speed_mask = speed > min_speed
    spatial_mask = (position >= reward_zone) & (position <= (track_length - reward_zone))
    return speed_mask & spatial_mask


def calculate_transient_rate_map(
    position: np.ndarray,
    transient_frames: np.ndarray,
    valid_mask: np.ndarray,
    bin_edges: np.ndarray,
    frame_rate: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate transient rate map (transients per second per bin)."""
    valid_transient_mask = valid_mask[transient_frames] if len(transient_frames) > 0 else np.array([], dtype=bool)
    valid_transient_frames = transient_frames[valid_transient_mask] if len(transient_frames) > 0 else np.array([])
    valid_transient_positions = position[valid_transient_frames] if len(valid_transient_frames) > 0 else np.array([])

    transient_counts, _ = np.histogram(valid_transient_positions, bins=bin_edges)

    valid_positions = position[valid_mask]
    occupancy_frames, _ = np.histogram(valid_positions, bins=bin_edges)
    occupancy_time = occupancy_frames / frame_rate

    with np.errstate(divide='ignore', invalid='ignore'):
        rate_map = np.where(occupancy_time > 0, transient_counts / occupancy_time, 0)

    return rate_map, occupancy_time, transient_counts


def smooth_and_normalize_place_field(
    rate_map: np.ndarray,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Gaussian smoothing and normalize place field."""
    smoothed = gaussian_filter1d(rate_map, sigma=sigma, mode='nearest')

    max_val = smoothed.max()
    normalized = smoothed / max_val if max_val > 0 else smoothed

    return smoothed, normalized


def calculate_mutual_information(
    transient_positions: np.ndarray,
    occupancy_time: np.ndarray,
    mi_bin_edges: np.ndarray,
) -> float:
    """Calculate mutual information between transients and location."""
    if len(transient_positions) == 0:
        return 0.0

    transient_counts, _ = np.histogram(transient_positions, bins=mi_bin_edges)

    total_transients = transient_counts.sum()
    total_time = occupancy_time.sum()

    if total_transients == 0 or total_time == 0:
        return 0.0

    overall_rate = total_transients / total_time
    n_bins = len(mi_bin_edges) - 1

    mi = 0.0
    for i in range(n_bins):
        if occupancy_time[i] > 0 and transient_counts[i] > 0:
            p_i = occupancy_time[i] / total_time
            rate_i = transient_counts[i] / occupancy_time[i]

            if rate_i > 0 and overall_rate > 0:
                mi += p_i * (rate_i / overall_rate) * np.log2(rate_i / overall_rate)

    return mi


def calculate_mi_with_shuffle_test(
    position: np.ndarray,
    transient_frames: np.ndarray,
    valid_mask: np.ndarray,
    mi_bin_edges: np.ndarray,
    frame_rate: float,
    n_shuffles: int,
) -> Tuple[float, float, np.ndarray]:
    """Calculate MI and perform shuffle test."""
    n_frames = len(position)

    valid_positions = position[valid_mask]
    occupancy_frames, _ = np.histogram(valid_positions, bins=mi_bin_edges)
    occupancy_time = occupancy_frames / frame_rate

    valid_transient_mask = valid_mask[transient_frames] if len(transient_frames) > 0 else np.array([], dtype=bool)
    valid_transient_frames = transient_frames[valid_transient_mask] if len(transient_frames) > 0 else np.array([])
    valid_transient_positions = position[valid_transient_frames] if len(valid_transient_frames) > 0 else np.array([])

    true_mi = calculate_mutual_information(valid_transient_positions, occupancy_time, mi_bin_edges)

    if len(transient_frames) == 0 or n_frames <= 1:
        return true_mi, 1.0, np.zeros(n_shuffles)

    # Vectorized shuffle
    shifts = np.random.randint(1, n_frames, size=n_shuffles)
    shuffled_frames_all = (transient_frames[np.newaxis, :] + shifts[:, np.newaxis]) % n_frames

    shuffle_mis = np.zeros(n_shuffles)
    for s in range(n_shuffles):
        shuffled_frames = shuffled_frames_all[s]
        shuffled_valid_mask = valid_mask[shuffled_frames]
        shuffled_valid_frames = shuffled_frames[shuffled_valid_mask]

        if len(shuffled_valid_frames) == 0:
            shuffle_mis[s] = 0.0
            continue

        shuffled_positions = position[shuffled_valid_frames]
        shuffle_mis[s] = calculate_mutual_information(shuffled_positions, occupancy_time, mi_bin_edges)

    p_value = (np.sum(shuffle_mis >= true_mi) + 1) / (n_shuffles + 1)

    return true_mi, p_value, shuffle_mis


def analyze_cell_place_field(
    position: np.ndarray,
    speed: np.ndarray,
    timestamps: np.ndarray,
    activity: np.ndarray,
    cell_sn: float,
    direction: str,
    mode: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyze place field for a single cell in a single direction."""
    params = config['analysis']
    frame_rate = config['cnmfe']['frame_rate']

    track_length = config['behavior']['track_length_cm']
    n_bins = params['n_bins']
    bin_edges = np.linspace(0, track_length, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_size = track_length / n_bins
    smoothing_sigma_bins = params['smoothing_sigma_cm'] / bin_size

    mi_bin_size = params['mi_bin_size_cm']
    n_mi_bins = int(track_length / mi_bin_size)
    mi_bin_edges = np.linspace(0, track_length, n_mi_bins + 1)

    valid_mask = apply_movement_and_spatial_filter(
        position, speed, params['min_speed_cm_s'], params['reward_zone_cm'], track_length
    )

    if mode == 'C':
        transient_frames = detect_transients_C_mode(
            activity, cell_sn, frame_rate,
            config['extraction']['temporal_window_ms'],
            config['extraction']['amplitude_threshold_sigma'],
        )
    else:
        transient_frames = detect_transients_S_mode(activity)

    rate_map, occupancy, transient_counts = calculate_transient_rate_map(
        position, transient_frames, valid_mask, bin_edges, frame_rate
    )

    smoothed_rate_map, normalized_place_field = smooth_and_normalize_place_field(
        rate_map, smoothing_sigma_bins
    )

    field_width = np.sum(normalized_place_field >= 0.5)
    peak_bin = np.argmax(normalized_place_field)
    peak_cm = bin_centers[peak_bin]

    total_weight = np.sum(normalized_place_field)
    centroid_cm = np.sum(normalized_place_field * bin_centers) / total_weight if total_weight > 0 else peak_cm

    true_mi, p_value, _ = calculate_mi_with_shuffle_test(
        position, transient_frames, valid_mask, mi_bin_edges, frame_rate, params['n_shuffles']
    )

    is_place_cell = p_value <= params['p_threshold']

    return {
        'rate_map': rate_map,
        'smoothed_rate_map': smoothed_rate_map,
        'normalized_place_field': normalized_place_field,
        'occupancy': occupancy,
        'transient_counts': transient_counts,
        'n_transients': len(transient_frames),
        'field_width': field_width,
        'field_width_cm': field_width * bin_size,
        'centroid_cm': centroid_cm,
        'peak_cm': peak_cm,
        'mutual_information': true_mi,
        'p_value': p_value,
        'is_place_cell': is_place_cell,
        'direction': direction,
    }


def analyze_all_cells(
    data: Dict[str, Any],
    config: Dict[str, Any],
    mode: str,
) -> Tuple[List[Dict], List[Dict]]:
    """Analyze all cells across all trials."""
    n_trials = len(data['Trial'])
    global_cell_ids = data['active_cell_ids']
    sn_values = data.get('active_cells_sn', np.ones(len(global_cell_ids)))
    frame_rate = config['cnmfe']['frame_rate']

    all_trial_results = []

    for i in range(n_trials):
        position = data['Position_x'][i]
        timestamps = data['TimeStamps'][i]
        activity = data['active_cells_Activity'][i]
        direction = data['Direction'][i]

        if not hasattr(position, '__len__'):
            position = np.array([position])
        if not hasattr(timestamps, '__len__'):
            timestamps = np.array([timestamps])
        if activity.ndim == 1:
            activity = activity.reshape(1, -1)

        speed = calculate_speed(position, timestamps)
        n_cells = activity.shape[1] if len(activity.shape) > 1 else 1

        cell_results = []
        for c in range(n_cells):
            cell_activity = activity[:, c] if n_cells > 1 else activity.flatten()
            cell_sn = sn_values[c] if c < len(sn_values) else 1.0

            result = analyze_cell_place_field(
                position, speed, timestamps, cell_activity,
                cell_sn, direction, mode, config
            )
            result['cell_id'] = global_cell_ids[c]
            result['cell_idx'] = c
            cell_results.append(result)

        all_trial_results.append({
            'trial_idx': data['Trial'][i],
            'direction': direction,
            'n_cells': n_cells,
            'cell_results': cell_results,
        })

        if (i + 1) % 10 == 0:
            print(f"    Processed trial {i+1}/{n_trials}")

    return all_trial_results


def aggregate_by_direction(
    all_trial_results: List[Dict],
    direction: str,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Aggregate results across trials for a specific direction."""
    dir_trials = [t for t in all_trial_results if t['direction'] == direction]

    if not dir_trials:
        return None

    n_cells = dir_trials[0]['n_cells']
    params = config['analysis']
    n_bins = params['n_bins']

    aggregated_cells = []
    for cell_idx in range(n_cells):
        transient_counts_total = np.zeros(n_bins)
        occupancy_total = np.zeros(n_bins)
        n_transients_total = 0
        mi_values = []
        p_values = []
        per_trial_field_width_cm = []
        per_trial_centroid_cm = []
        per_trial_peak_cm = []

        for trial in dir_trials:
            cell_result = trial['cell_results'][cell_idx]
            transient_counts_total += cell_result['transient_counts']
            occupancy_total += cell_result['occupancy']
            n_transients_total += cell_result['n_transients']
            mi_values.append(cell_result['mutual_information'])
            p_values.append(cell_result['p_value'])
            per_trial_field_width_cm.append(cell_result['field_width_cm'])
            per_trial_centroid_cm.append(cell_result['centroid_cm'])
            per_trial_peak_cm.append(cell_result['peak_cm'])

        with np.errstate(divide='ignore', invalid='ignore'):
            combined_rate_map = np.where(occupancy_total > 0, transient_counts_total / occupancy_total, 0)

        bin_size = config['behavior']['track_length_cm'] / n_bins
        smoothing_sigma_bins = params['smoothing_sigma_cm'] / bin_size
        smoothed, normalized = smooth_and_normalize_place_field(combined_rate_map, smoothing_sigma_bins)

        field_width = np.sum(normalized >= 0.5)
        min_p = np.min(p_values)
        is_place_cell = min_p <= params['p_threshold']

        bin_centers = np.linspace(0, config['behavior']['track_length_cm'], n_bins + 1)
        bin_centers = (bin_centers[:-1] + bin_centers[1:]) / 2
        peak_bin = np.argmax(normalized)
        peak_cm = bin_centers[peak_bin]
        total_weight = np.sum(normalized)
        centroid_cm = np.sum(normalized * bin_centers) / total_weight if total_weight > 0 else peak_cm

        aggregated_cells.append({
            'cell_id': dir_trials[0]['cell_results'][cell_idx]['cell_id'],
            'cell_idx': cell_idx,
            'combined_normalized': normalized,
            'combined_rate_map': combined_rate_map,
            'field_width': field_width,
            'field_width_cm': field_width * bin_size,
            'centroid_cm': centroid_cm,
            'peak_cm': peak_cm,
            'mean_mi': np.mean(mi_values),
            'min_p_value': min_p,
            'is_place_cell': is_place_cell,
            'n_trials': len(dir_trials),
            'total_transients': n_transients_total,
            'per_trial_field_width_cm': np.array(per_trial_field_width_cm),
            'per_trial_centroid_cm': np.array(per_trial_centroid_cm),
            'per_trial_peak_cm': np.array(per_trial_peak_cm),
        })

    return {
        'direction': direction,
        'n_trials': len(dir_trials),
        'n_cells': n_cells,
        'cell_results': aggregated_cells,
        'n_place_cells': sum(1 for c in aggregated_cells if c['is_place_cell']),
    }


def aggregate_session_results(
    all_trial_results: List[Dict],
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Aggregate results across ALL trials (regardless of direction) for session-wide firing rate.

    Returns firing count (number of Ca2+ activations) per position bin for each neuron
    across the entire session.
    """
    if not all_trial_results:
        return None

    n_cells = all_trial_results[0]['n_cells']
    params = config['analysis']
    n_bins = params['n_bins']
    track_length = config['behavior']['track_length_cm']
    bin_size = track_length / n_bins
    smoothing_sigma_bins = params['smoothing_sigma_cm'] / bin_size

    session_cells = []
    for cell_idx in range(n_cells):
        transient_counts_total = np.zeros(n_bins)
        trial_counts = np.zeros(n_bins)
        occupancy_total = np.zeros(n_bins)
        n_transients_total = 0

        for trial in all_trial_results:
            cell_result = trial['cell_results'][cell_idx]
            transient_counts_total += cell_result['transient_counts']
            occupancy_total += cell_result['occupancy']
            n_transients_total += cell_result['n_transients']

            trial_activity = (cell_result['transient_counts'] > 0).astype(float)
            trial_counts += trial_activity

        with np.errstate(divide='ignore', invalid='ignore'):
            firing_rate = np.where(occupancy_total > 0,
                                   transient_counts_total / occupancy_total, 0)

        smoothed_rate, normalized_rate = smooth_and_normalize_place_field(firing_rate, smoothing_sigma_bins)

        n_trials = len(all_trial_results)
        trial_counts_normalized = trial_counts / n_trials if n_trials > 0 else trial_counts

        session_cells.append({
            'cell_id': all_trial_results[0]['cell_results'][cell_idx].get('cell_id', cell_idx),
            'cell_idx': cell_idx,
            'firing_counts': transient_counts_total,
            'trial_counts': trial_counts,
            'trial_counts_normalized': trial_counts_normalized,
            'occupancy': occupancy_total,
            'firing_rate': firing_rate,
            'smoothed_firing_rate': smoothed_rate,
            'normalized_firing_rate': normalized_rate,
            'total_transients': n_transients_total,
        })

    return {
        'n_trials': len(all_trial_results),
        'n_cells': n_cells,
        'cell_results': session_cells,
    }


def plot_place_cell_heatmap(
    results: Dict[str, Any],
    output_dir: Path,
    mode: str,
    config: Dict[str, Any],
) -> None:
    """Plot heatmap of place cells sorted by peak location."""
    direction = results['direction']
    cell_results = results['cell_results']

    place_cells = [c for c in cell_results if c['is_place_cell']]
    n_place_cells = len(place_cells)

    if n_place_cells == 0:
        return

    place_fields = np.array([c['combined_normalized'] for c in place_cells])
    peak_locations = np.argmax(place_fields, axis=1)
    sort_idx = np.argsort(peak_locations)
    place_fields_sorted = place_fields[sort_idx]

    track_length = config['behavior']['track_length_cm']
    reward_zone = config['analysis']['reward_zone_cm']

    fig, ax = plt.subplots(figsize=(12, max(8, n_place_cells * 0.15)))

    im = ax.imshow(place_fields_sorted, aspect='auto', cmap='jet',
                   extent=[0, track_length, n_place_cells, 0],
                   interpolation='nearest', vmin=0, vmax=1)

    ax.axvline(x=reward_zone, color='cyan', linestyle='--', alpha=0.7)
    ax.axvline(x=track_length - reward_zone, color='cyan', linestyle='--', alpha=0.7)

    ax.set_xlabel('Position (cm)', fontsize=12)
    ax.set_ylabel('Place Cell (sorted by peak)', fontsize=12)
    ax.set_title(f'Place Cells - Direction: {direction} | {n_place_cells}/{results["n_cells"]} | Mode: {mode}',
                 fontsize=14)

    plt.colorbar(im, ax=ax, label='Normalized Activity')
    plt.tight_layout()

    fig_path = output_dir / f'place_cells_heatmap_{direction}_{mode}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_trial_place_fields(
    trial_result: Dict[str, Any],
    output_dir: Path,
    mode: str,
    config: Dict[str, Any],
) -> None:
    """Plot place field heatmap for a single trial (all cells)."""
    trial_idx = trial_result['trial_idx']
    direction = trial_result['direction']
    cell_results = trial_result['cell_results']
    n_cells = len(cell_results)

    if n_cells == 0:
        return

    track_length = config['behavior']['track_length_cm']
    reward_zone = config['analysis']['reward_zone_cm']

    place_fields = np.array([c['normalized_place_field'] for c in cell_results])

    peak_locations = np.argmax(place_fields, axis=1)
    sort_idx = np.argsort(peak_locations)
    place_fields_sorted = place_fields[sort_idx]

    fig, ax = plt.subplots(figsize=(12, max(6, n_cells * 0.08)))

    im = ax.imshow(place_fields_sorted, aspect='auto', cmap='jet',
                   extent=[0, track_length, n_cells, 0],
                   interpolation='nearest', vmin=0, vmax=1)

    ax.axvline(x=reward_zone, color='cyan', linestyle='--', alpha=0.7)
    ax.axvline(x=track_length - reward_zone, color='cyan', linestyle='--', alpha=0.7)

    ax.set_xlabel('Position (cm)', fontsize=12)
    ax.set_ylabel('Cell (sorted by peak)', fontsize=12)
    ax.set_title(f'Trial {trial_idx} | Direction: {direction} | {n_cells} Cells | Mode: {mode}',
                 fontsize=14)

    plt.colorbar(im, ax=ax, label='Normalized Activity')
    plt.tight_layout()

    fig_path = output_dir / f'trial_{trial_idx:03d}_place_fields.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_trial_behavior(
    trial_data: Dict[str, Any],
    trial_result: Dict[str, Any],
    output_dir: Path,
    mode: str,
    config: Dict[str, Any],
) -> None:
    """Plot behavioral data (position, speed) and neural activity for a single trial."""
    trial_idx = trial_result['trial_idx']
    direction = trial_result['direction']

    position = trial_data['position']
    timestamps = trial_data['timestamps']
    activity = trial_data['activity']

    speed = calculate_speed(position, timestamps)

    params = config['analysis']
    track_length = config['behavior']['track_length_cm']
    valid_mask = apply_movement_and_spatial_filter(
        position, speed, params['min_speed_cm_s'], params['reward_zone_cm'], track_length
    )

    t_rel = timestamps - timestamps[0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: Position
    ax = axes[0]
    ax.plot(t_rel, position, 'b-', linewidth=1, alpha=0.8)

    invalid_mask = ~valid_mask
    if np.any(invalid_mask):
        ax.scatter(t_rel[invalid_mask], position[invalid_mask],
                   c='red', s=2, alpha=0.3, label='Excluded')

    ax.axhline(y=params['reward_zone_cm'], color='cyan', linestyle='--', alpha=0.7, label='Reward zone')
    ax.axhline(y=track_length - params['reward_zone_cm'], color='cyan', linestyle='--', alpha=0.7)

    ax.set_ylabel('Position (cm)', fontsize=11)
    ax.set_ylim(0, track_length)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f'Trial {trial_idx} | Direction: {direction} | Mode: {mode}', fontsize=14)

    # Panel 2: Speed
    ax = axes[1]
    ax.plot(t_rel, speed, 'g-', linewidth=1, alpha=0.8)
    ax.axhline(y=params['min_speed_cm_s'], color='red', linestyle='--', alpha=0.7,
               label=f'Min speed ({params["min_speed_cm_s"]} cm/s)')

    ax.set_ylabel('Speed (cm/s)', fontsize=11)
    ax.set_ylim(0, max(speed.max() * 1.1, params['min_speed_cm_s'] * 2))
    ax.legend(loc='upper right', fontsize=9)

    # Panel 3: Neural Activity Heatmap
    ax = axes[2]
    n_cells = activity.shape[1] if len(activity.shape) > 1 else 1

    if n_cells > 1:
        activity_norm = activity.copy().astype(float)
        for i in range(n_cells):
            cell_max = activity[:, i].max()
            if cell_max > 0:
                activity_norm[:, i] = activity[:, i] / cell_max

        im = ax.imshow(activity_norm.T, aspect='auto', cmap='jet',
                       extent=[t_rel[0], t_rel[-1], n_cells, 0],
                       interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Normalized Activity')
    else:
        ax.plot(t_rel, activity.flatten(), 'k-', linewidth=1)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Cell', fontsize=11)

    plt.tight_layout()
    fig_path = output_dir / f'trial_{trial_idx:03d}_behavior.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_summary_statistics(
    left_results: Optional[Dict],
    right_results: Optional[Dict],
    output_dir: Path,
    mode: str,
    config: Dict[str, Any],
) -> None:
    """Plot summary statistics comparing left and right directions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Number of place cells
    ax = axes[0, 0]
    categories = ['Left', 'Right']
    n_place_cells = [
        left_results['n_place_cells'] if left_results else 0,
        right_results['n_place_cells'] if right_results else 0
    ]
    n_total = [
        left_results['n_cells'] if left_results else 0,
        right_results['n_cells'] if right_results else 0
    ]

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, n_place_cells, width, label='Place Cells', color='green', alpha=0.7)
    ax.bar(x + width/2, n_total, width, label='All Cells', color='gray', alpha=0.7)
    ax.set_ylabel('Number of Cells')
    ax.set_title('Place Cells by Direction')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Field width distribution
    ax = axes[0, 1]
    if left_results:
        left_widths = [c['field_width_cm'] for c in left_results['cell_results'] if c['is_place_cell']]
        if left_widths:
            ax.hist(left_widths, bins=20, alpha=0.5, label='Left', color='red')
    if right_results:
        right_widths = [c['field_width_cm'] for c in right_results['cell_results'] if c['is_place_cell']]
        if right_widths:
            ax.hist(right_widths, bins=20, alpha=0.5, label='Right', color='blue')
    ax.set_xlabel('Place Field Width (cm)')
    ax.set_ylabel('Count')
    ax.set_title('Place Field Width Distribution')
    ax.legend()

    # MI distribution
    ax = axes[1, 0]
    if left_results:
        left_mi = [c['mean_mi'] for c in left_results['cell_results'] if c['is_place_cell']]
        if left_mi:
            ax.hist(left_mi, bins=20, alpha=0.5, label='Left', color='red')
    if right_results:
        right_mi = [c['mean_mi'] for c in right_results['cell_results'] if c['is_place_cell']]
        if right_mi:
            ax.hist(right_mi, bins=20, alpha=0.5, label='Right', color='blue')
    ax.set_xlabel('Mutual Information (bits)')
    ax.set_ylabel('Count')
    ax.set_title('MI Distribution (Place Cells)')
    ax.legend()

    # P-value distribution
    ax = axes[1, 1]
    p_threshold = config['analysis']['p_threshold']
    if left_results:
        left_p = [c['min_p_value'] for c in left_results['cell_results']]
        ax.hist(left_p, bins=50, alpha=0.5, label='Left', color='red')
    if right_results:
        right_p = [c['min_p_value'] for c in right_results['cell_results']]
        ax.hist(right_p, bins=50, alpha=0.5, label='Right', color='blue')
    ax.axvline(x=p_threshold, color='black', linestyle='--', label=f'P = {p_threshold}')
    ax.set_xlabel('P-value')
    ax.set_ylabel('Count')
    ax.set_title('P-value Distribution (All Cells)')
    ax.legend()

    plt.suptitle(f'Place Cell Analysis Summary | Mode: {mode}', fontsize=16)
    plt.tight_layout()

    fig_path = output_dir / f'place_cell_summary_{mode}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_session_firing_rate_heatmap(
    session_results: Optional[Dict[str, Any]],
    output_dir: Path,
    mode: str,
    config: Dict[str, Any],
) -> Optional[np.ndarray]:
    """
    Plot session-wide firing rate heatmap showing firing counts per position for each cell.

    Generates three plots:
    1. Raw firing counts (integer number of transients)
    2. Firing rate (transients per second, normalized for occupancy)
    3. Trial counts (number of trials with activity per bin)

    Returns sort_idx for consistent ordering in other plots.
    """
    if session_results is None:
        return None

    cell_results = session_results['cell_results']
    n_cells = len(cell_results)
    n_trials = session_results['n_trials']

    if n_cells == 0:
        return None

    track_length = config['behavior']['track_length_cm']
    reward_zone = config['analysis']['reward_zone_cm']

    # --- Plot 1: Firing Counts ---
    firing_counts = np.array([c['firing_counts'] for c in cell_results])

    peak_locations = np.argmax(firing_counts, axis=1)
    sort_idx = np.argsort(peak_locations)
    firing_counts_sorted = firing_counts[sort_idx]

    fig, ax = plt.subplots(figsize=(12, max(6, n_cells * 0.08)))

    im = ax.imshow(firing_counts_sorted, aspect='auto', cmap='jet',
                   extent=[0, track_length, n_cells, 0],
                   interpolation='nearest')

    ax.axvline(x=reward_zone, color='cyan', linestyle='--', alpha=0.7)
    ax.axvline(x=track_length - reward_zone, color='cyan', linestyle='--', alpha=0.7)

    ax.set_xlabel('Position (cm)', fontsize=12)
    ax.set_ylabel('Cell (sorted by peak)', fontsize=12)
    ax.set_title(f'Session-Wide Firing Counts | {n_cells} Cells | Mode: {mode}\n'
                 f'(Total transients per position bin across {n_trials} trials)',
                 fontsize=14)

    plt.colorbar(im, ax=ax, label='Transient Count')
    plt.tight_layout()

    fig_path = output_dir / f'session_firing_counts_{mode}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 2: Firing Rate (normalized) ---
    normalized_rates = np.array([c['normalized_firing_rate'] for c in cell_results])
    normalized_rates_sorted = normalized_rates[sort_idx]

    fig, ax = plt.subplots(figsize=(12, max(6, n_cells * 0.08)))

    im = ax.imshow(normalized_rates_sorted, aspect='auto', cmap='jet',
                   extent=[0, track_length, n_cells, 0],
                   interpolation='nearest', vmin=0, vmax=1)

    ax.axvline(x=reward_zone, color='cyan', linestyle='--', alpha=0.7)
    ax.axvline(x=track_length - reward_zone, color='cyan', linestyle='--', alpha=0.7)

    ax.set_xlabel('Position (cm)', fontsize=12)
    ax.set_ylabel('Cell (sorted by peak)', fontsize=12)
    ax.set_title(f'Session-Wide Firing Rate (Normalized) | {n_cells} Cells | Mode: {mode}\n'
                 f'(Smoothed & normalized firing rate across {n_trials} trials)',
                 fontsize=14)

    plt.colorbar(im, ax=ax, label='Normalized Firing Rate')
    plt.tight_layout()

    fig_path = output_dir / f'session_firing_rate_normalized_{mode}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 3: Trial Counts ---
    trial_counts = np.array([c['trial_counts'] for c in cell_results])
    trial_counts_sorted = trial_counts[sort_idx]

    fig, ax = plt.subplots(figsize=(12, max(6, n_cells * 0.08)))

    im = ax.imshow(trial_counts_sorted, aspect='auto', cmap='jet',
                   extent=[0, track_length, n_cells, 0],
                   interpolation='nearest', vmin=0, vmax=n_trials)

    ax.axvline(x=reward_zone, color='cyan', linestyle='--', alpha=0.7)
    ax.axvline(x=track_length - reward_zone, color='cyan', linestyle='--', alpha=0.7)

    ax.set_xlabel('Position (cm)', fontsize=12)
    ax.set_ylabel('Cell (sorted by peak)', fontsize=12)
    ax.set_title(f'Session-Wide Trial Counts | {n_cells} Cells | Mode: {mode}\n'
                 f'(Number of trials with activity per bin, max = {n_trials})',
                 fontsize=14)

    plt.colorbar(im, ax=ax, label='Trial Count')
    plt.tight_layout()

    fig_path = output_dir / f'session_trial_counts_{mode}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    return sort_idx


def plot_combined_direction_heatmaps(
    left_results: Optional[Dict[str, Any]],
    right_results: Optional[Dict[str, Any]],
    output_dir: Path,
    mode: str,
    config: Dict[str, Any],
) -> None:
    """
    Plot combined L/R heatmaps with joint normalization.

    Creates 2 figures:
    1. L-place cells only: Left (sorted by peak) | Right (same L-place cells, same order)
    2. R-place cells only: Right (sorted by peak) | Left (same R-place cells, same order)

    For each neuron, normalization is done jointly across both directions.
    """
    if left_results is None or right_results is None:
        print("  Cannot create combined heatmaps: missing L or R results")
        return

    track_length = config['behavior']['track_length_cm']
    reward_zone = config['analysis']['reward_zone_cm']
    n_bins = config['analysis']['n_bins']
    bin_size = track_length / n_bins
    smoothing_sigma_bins = config['analysis']['smoothing_sigma_cm'] / bin_size

    n_total_cells = left_results['n_cells']

    left_is_place_cell = np.array([c['is_place_cell'] for c in left_results['cell_results']])
    right_is_place_cell = np.array([c['is_place_cell'] for c in right_results['cell_results']])

    # === Figure 1: L-place cells sorted by Left peak ===
    left_pc_indices = np.where(left_is_place_cell)[0]
    n_left_pc = len(left_pc_indices)

    if n_left_pc > 0:
        left_rate_maps_L = np.array([left_results['cell_results'][i]['combined_rate_map'] for i in left_pc_indices])
        right_rate_maps_L = np.array([right_results['cell_results'][i]['combined_rate_map'] for i in left_pc_indices])

        left_smoothed_L = np.array([gaussian_filter1d(rm, sigma=smoothing_sigma_bins, mode='nearest')
                                    for rm in left_rate_maps_L])
        right_smoothed_L = np.array([gaussian_filter1d(rm, sigma=smoothing_sigma_bins, mode='nearest')
                                     for rm in right_rate_maps_L])

        left_normalized_L = np.zeros_like(left_smoothed_L)
        right_normalized_L = np.zeros_like(right_smoothed_L)

        for i in range(n_left_pc):
            joint_max = max(left_smoothed_L[i].max(), right_smoothed_L[i].max())
            if joint_max > 0:
                left_normalized_L[i] = left_smoothed_L[i] / joint_max
                right_normalized_L[i] = right_smoothed_L[i] / joint_max

        left_peak_locations = np.argmax(left_normalized_L, axis=1)
        left_sort_idx = np.argsort(left_peak_locations)

        left_sorted = left_normalized_L[left_sort_idx]
        right_sorted_by_left = right_normalized_L[left_sort_idx]

        fig, axes = plt.subplots(1, 2, figsize=(20, max(8, n_left_pc * 0.15)))

        ax = axes[0]
        im = ax.imshow(left_sorted, aspect='auto', cmap='jet',
                       extent=[0, track_length, n_left_pc, 0],
                       interpolation='nearest', vmin=0, vmax=1)

        ax.axvline(x=reward_zone, color='cyan', linestyle='--', alpha=0.7)
        ax.axvline(x=track_length - reward_zone, color='cyan', linestyle='--', alpha=0.7)
        ax.set_xlabel('Position (cm)', fontsize=12)
        ax.set_ylabel('L-Place Cell (sorted by Left peak)', fontsize=12)
        ax.set_title(f'LEFT Direction (sorted) | {n_left_pc} L-Place Cells', fontsize=13)

        ax = axes[1]
        ax.imshow(right_sorted_by_left, aspect='auto', cmap='jet',
                  extent=[0, track_length, n_left_pc, 0],
                  interpolation='nearest', vmin=0, vmax=1)

        ax.axvline(x=reward_zone, color='cyan', linestyle='--', alpha=0.7)
        ax.axvline(x=track_length - reward_zone, color='cyan', linestyle='--', alpha=0.7)
        ax.set_xlabel('Position (cm)', fontsize=12)
        ax.set_ylabel('')
        ax.set_title(f'RIGHT Direction (same cells) | {n_left_pc} L-Place Cells', fontsize=13)

        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Jointly Normalized Activity')

        plt.suptitle(f'L-Place Cells: Left vs Right Direction | Mode: {mode}\n'
                     f'Joint normalization per neuron | {n_left_pc}/{n_total_cells} cells', fontsize=14)

        fig_path = output_dir / f'combined_heatmap_sorted_by_L_{mode}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

    # === Figure 2: R-place cells sorted by Right peak ===
    right_pc_indices = np.where(right_is_place_cell)[0]
    n_right_pc = len(right_pc_indices)

    if n_right_pc > 0:
        left_rate_maps_R = np.array([left_results['cell_results'][i]['combined_rate_map'] for i in right_pc_indices])
        right_rate_maps_R = np.array([right_results['cell_results'][i]['combined_rate_map'] for i in right_pc_indices])

        left_smoothed_R = np.array([gaussian_filter1d(rm, sigma=smoothing_sigma_bins, mode='nearest')
                                    for rm in left_rate_maps_R])
        right_smoothed_R = np.array([gaussian_filter1d(rm, sigma=smoothing_sigma_bins, mode='nearest')
                                     for rm in right_rate_maps_R])

        left_normalized_R = np.zeros_like(left_smoothed_R)
        right_normalized_R = np.zeros_like(right_smoothed_R)

        for i in range(n_right_pc):
            joint_max = max(left_smoothed_R[i].max(), right_smoothed_R[i].max())
            if joint_max > 0:
                left_normalized_R[i] = left_smoothed_R[i] / joint_max
                right_normalized_R[i] = right_smoothed_R[i] / joint_max

        right_peak_locations = np.argmax(right_normalized_R, axis=1)
        right_sort_idx = np.argsort(right_peak_locations)

        right_sorted = right_normalized_R[right_sort_idx]
        left_sorted_by_right = left_normalized_R[right_sort_idx]

        fig, axes = plt.subplots(1, 2, figsize=(20, max(8, n_right_pc * 0.15)))

        ax = axes[0]
        ax.imshow(left_sorted_by_right, aspect='auto', cmap='jet',
                  extent=[0, track_length, n_right_pc, 0],
                  interpolation='nearest', vmin=0, vmax=1)

        ax.axvline(x=reward_zone, color='cyan', linestyle='--', alpha=0.7)
        ax.axvline(x=track_length - reward_zone, color='cyan', linestyle='--', alpha=0.7)
        ax.set_xlabel('Position (cm)', fontsize=12)
        ax.set_ylabel('R-Place Cell (sorted by Right peak)', fontsize=12)
        ax.set_title(f'LEFT Direction (same cells) | {n_right_pc} R-Place Cells', fontsize=13)

        ax = axes[1]
        im = ax.imshow(right_sorted, aspect='auto', cmap='jet',
                       extent=[0, track_length, n_right_pc, 0],
                       interpolation='nearest', vmin=0, vmax=1)

        ax.axvline(x=reward_zone, color='cyan', linestyle='--', alpha=0.7)
        ax.axvline(x=track_length - reward_zone, color='cyan', linestyle='--', alpha=0.7)
        ax.set_xlabel('Position (cm)', fontsize=12)
        ax.set_ylabel('')
        ax.set_title(f'RIGHT Direction (sorted) | {n_right_pc} R-Place Cells', fontsize=13)

        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Jointly Normalized Activity')

        plt.suptitle(f'R-Place Cells: Left vs Right Direction | Mode: {mode}\n'
                     f'Joint normalization per neuron | {n_right_pc}/{n_total_cells} cells', fontsize=14)

        fig_path = output_dir / f'combined_heatmap_sorted_by_R_{mode}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()


def save_results_mat(
    left_results: Optional[Dict],
    right_results: Optional[Dict],
    output_dir: Path,
    mode: str,
    config: Dict[str, Any],
    session_results: Optional[Dict] = None,
) -> Path:
    """Save analysis results to MAT file."""
    def extract_cell_data(results):
        if results is None:
            return {}

        cell_results = results['cell_results']
        n_cells = len(cell_results)

        return {
            'direction': results['direction'],
            'n_trials': results['n_trials'],
            'n_cells': n_cells,
            'n_place_cells': results['n_place_cells'],
            'cell_ids': np.array([c['cell_id'] for c in cell_results]),
            'is_place_cell': np.array([c['is_place_cell'] for c in cell_results]),
            'place_fields': np.array([c['combined_normalized'] for c in cell_results]),
            'rate_maps': np.array([c['combined_rate_map'] for c in cell_results]),
            'field_widths_cm': np.array([c['field_width_cm'] for c in cell_results]),
            'centroids_cm': np.array([c['centroid_cm'] for c in cell_results]),
            'peaks_cm': np.array([c['peak_cm'] for c in cell_results]),
            'mutual_information': np.array([c['mean_mi'] for c in cell_results]),
            'p_values': np.array([c['min_p_value'] for c in cell_results]),
        }

    def extract_session_data(results):
        if results is None:
            return {}

        cell_results = results['cell_results']
        return {
            'n_trials': results['n_trials'],
            'n_cells': results['n_cells'],
            'cell_ids': np.array([c['cell_id'] for c in cell_results]),
            'firing_counts': np.array([c['firing_counts'] for c in cell_results]),
            'trial_counts': np.array([c['trial_counts'] for c in cell_results]),
            'trial_counts_normalized': np.array([c['trial_counts_normalized'] for c in cell_results]),
            'occupancy': np.array([c['occupancy'] for c in cell_results]),
            'firing_rate': np.array([c['firing_rate'] for c in cell_results]),
            'normalized_firing_rate': np.array([c['normalized_firing_rate'] for c in cell_results]),
            'total_transients': np.array([c['total_transients'] for c in cell_results]),
        }

    params = config['analysis']
    track_length = config['behavior']['track_length_cm']
    n_bins = params['n_bins']
    bin_edges = np.linspace(0, track_length, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate place cell statistics
    place_cell_stats = {}
    if left_results and right_results:
        left_pc = np.array([c['is_place_cell'] for c in left_results['cell_results']])
        right_pc = np.array([c['is_place_cell'] for c in right_results['cell_results']])

        place_cell_stats = {
            'only_left': int(np.sum(left_pc & ~right_pc)),
            'only_right': int(np.sum(~left_pc & right_pc)),
            'both': int(np.sum(left_pc & right_pc)),
            'total_union': int(np.sum(left_pc | right_pc)),
            'n_active_cells': len(left_pc),
        }

    output_data = {
        'left': extract_cell_data(left_results),
        'right': extract_cell_data(right_results),
        'session': extract_session_data(session_results),
        'place_cell_statistics': place_cell_stats,
        'bin_centers': bin_centers,
        'bin_edges': bin_edges,
        'parameters': {
            'track_length_cm': track_length,
            'n_bins': n_bins,
            'smoothing_sigma_cm': params['smoothing_sigma_cm'],
            'min_speed_cm_s': params['min_speed_cm_s'],
            'reward_zone_cm': params['reward_zone_cm'],
            'mi_bin_size_cm': params['mi_bin_size_cm'],
            'n_shuffles': params['n_shuffles'],
            'p_threshold': params['p_threshold'],
            'mode': mode,
        },
    }

    mat_path = output_dir / f'place_cell_results_{mode}.mat'
    save_mat_file(str(mat_path), output_data)

    return mat_path


def analyze_place_fields(
    config: Dict[str, Any],
    input_mat_path: Optional[str] = None,
    mode: str = 'S',
) -> Path:
    """
    Main function to run place field analysis.

    Args:
        config: Configuration dictionary
        input_mat_path: Path to downstream MAT file (Pipeline E output)
        mode: 'C' for calcium traces, 'S' for deconvolved spikes

    Returns:
        Path to output directory
    """
    experiment_name = config['experiment']['name']

    print(f"\n{'='*60}")
    print(f"PIPELINE F: PLACE FIELD ANALYSIS")
    print(f"{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Mode: {mode}")

    # Load input data
    if input_mat_path is None:
        output_dir = Path(config['experiment']['output_dir'])
        mat_files = list(output_dir.glob(f"{experiment_name}*downstream_{mode}.mat"))
        if mat_files:
            input_mat_path = str(sorted(mat_files)[-1])
        else:
            raise FileNotFoundError(f"No downstream MAT file found for mode {mode}")

    print(f"\nLoading: {input_mat_path}")
    data = load_mat_file(input_mat_path)

    n_trials = len(data['Trial'])
    n_cells = data['n_active_cells']
    print(f"  Trials: {n_trials}")
    print(f"  Active cells: {n_cells}")

    # Create output directory
    output_dir = get_output_path(config, '', f'place_field_analysis/{mode}_mode')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze all cells and collect trial data
    print(f"\n  Analyzing place fields...")
    all_trial_results = analyze_all_cells(data, config, mode)

    # Collect trial data for behavior plotting
    all_trial_data = []
    for i in range(n_trials):
        position = data['Position_x'][i]
        timestamps = data['TimeStamps'][i]
        activity = data['active_cells_Activity'][i]

        if not hasattr(position, '__len__'):
            position = np.array([position])
        if not hasattr(timestamps, '__len__'):
            timestamps = np.array([timestamps])
        if activity.ndim == 1:
            activity = activity.reshape(1, -1)

        all_trial_data.append({
            'position': position,
            'timestamps': timestamps,
            'activity': activity,
        })

    # Aggregate by direction
    print(f"\n  Aggregating by direction...")
    left_results = aggregate_by_direction(all_trial_results, 'L', config)
    right_results = aggregate_by_direction(all_trial_results, 'R', config)

    if left_results:
        print(f"    Left: {left_results['n_place_cells']}/{left_results['n_cells']} place cells ({left_results['n_trials']} trials)")
    if right_results:
        print(f"    Right: {right_results['n_place_cells']}/{right_results['n_cells']} place cells ({right_results['n_trials']} trials)")

    # Aggregate session-wide results
    print(f"\n  Aggregating session-wide firing rate...")
    session_results = aggregate_session_results(all_trial_results, config)
    if session_results:
        total_transients = sum(c['total_transients'] for c in session_results['cell_results'])
        print(f"    Total transients: {total_transients}")

    # Generate plots
    print(f"\n  Generating plots...")

    # --- Per-trial figures ---
    trials_dir = output_dir / 'trials'
    trials_dir.mkdir(parents=True, exist_ok=True)

    print(f"    Generating per-trial figures...")
    for i in range(n_trials):
        plot_trial_place_fields(all_trial_results[i], trials_dir, mode, config)
        plot_trial_behavior(all_trial_data[i], all_trial_results[i], trials_dir, mode, config)

        if (i + 1) % 20 == 0:
            print(f"      Trial figures: {i+1}/{n_trials}")

    print(f"    Generated {n_trials} trial figures in {trials_dir}")

    # --- Direction-aggregated figures ---
    if left_results:
        plot_place_cell_heatmap(left_results, output_dir, mode, config)
        print(f"    Generated Left direction heatmap")
    if right_results:
        plot_place_cell_heatmap(right_results, output_dir, mode, config)
        print(f"    Generated Right direction heatmap")

    # --- Combined L/R heatmaps ---
    plot_combined_direction_heatmaps(left_results, right_results, output_dir, mode, config)
    print(f"    Generated combined L/R heatmaps")

    # --- Session-wide firing rate heatmaps ---
    if session_results:
        plot_session_firing_rate_heatmap(session_results, output_dir, mode, config)
        print(f"    Generated session-wide firing rate heatmaps")

    plot_summary_statistics(left_results, right_results, output_dir, mode, config)
    print(f"    Generated summary statistics")

    # Save results
    print(f"\n  Saving results...")
    mat_path = save_results_mat(left_results, right_results, output_dir, mode, config, session_results)

    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Results MAT: {mat_path.name}")
    print(f"  Per-trial figures: {trials_dir}")
    print(f"{'='*60}")

    return output_dir
