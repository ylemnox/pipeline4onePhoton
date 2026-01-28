"""
Pipeline E: Active cell extraction and trial alignment.

Performs:
1. Good cell filtering based on CNMF-E evaluation (idx_components)
2. Duplicate cell removal (spatial proximity + correlation)
3. Global active cell detection (C or S mode)
4. Trial-by-trial data extraction

Output: Downstream MAT file with active cells and trial-aligned data.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

from ..config import get_output_path
from ..utils.io_utils import load_mat_file, save_mat_file


def filter_good_cells(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter to keep only good cells based on idx_components.

    Args:
        data: Data dictionary with C, S, idx_good, etc.

    Returns:
        Filtered data dictionary
    """
    idx_good = data['idx_good']
    n_good = len(idx_good)
    print(f"  Filtering good cells: {n_good} out of {data['C'].shape[1]}")

    return {
        'C': data['C'][:, idx_good],
        'S': data['S'][:, idx_good],
        'sn': data['sn'][idx_good],
        'SNR': data['SNR'][idx_good],
        'centroids': data['centroids'][idx_good],
        'original_indices': idx_good,
        'trial_times': data['trial_times'],
        'trial_direction': data['trial_direction'],
        'vr_time': data['vr_time'],
        'vr_position_x': data['vr_position_x'],
        'calcium_time': data['calcium_time'],
        'frame_rate': data['frame_rate'],
    }


def find_neighbors(centroids: np.ndarray, distance_threshold_px: float) -> list:
    """
    Find pairs of neurons within the specified distance threshold.

    Args:
        centroids: Neuron centroid positions (n_neurons, 2)
        distance_threshold_px: Distance threshold in pixels

    Returns:
        List of neighbor pairs (i, j) where i < j
    """
    n_neurons = len(centroids)
    distances = cdist(centroids, centroids)

    neighbor_pairs = []
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            if distances[i, j] <= distance_threshold_px:
                neighbor_pairs.append((i, j))

    return neighbor_pairs


def remove_duplicates_global(
    data: Dict[str, Any],
    neighbor_distance_um: float,
    pixel_size_um: float,
    correlation_threshold: float,
) -> Dict[str, Any]:
    """
    Remove duplicate/ghost cells globally.

    For neighbor pairs with Pearson correlation > threshold across the full
    session, remove the cell with lower SNR.

    Args:
        data: Filtered data dictionary
        neighbor_distance_um: Distance threshold in micrometers
        pixel_size_um: Pixel size in micrometers
        correlation_threshold: Correlation threshold for duplicates

    Returns:
        Data dictionary with duplicates removed
    """
    print(f"\n  Global duplicate removal:")
    print(f"    Neighbor distance: {neighbor_distance_um} um")
    print(f"    Correlation threshold: {correlation_threshold}")

    neighbor_distance_px = neighbor_distance_um / pixel_size_um

    C = data['C']
    n_neurons = C.shape[1]
    keep_mask = np.ones(n_neurons, dtype=bool)

    neighbor_pairs = find_neighbors(data['centroids'], neighbor_distance_px)
    print(f"    Found {len(neighbor_pairs)} neighbor pairs")

    cells_removed = 0
    for i, j in neighbor_pairs:
        if not keep_mask[i] or not keep_mask[j]:
            continue

        r, _ = pearsonr(C[:, i], C[:, j])

        if r > correlation_threshold:
            if data['SNR'][i] < data['SNR'][j]:
                keep_mask[i] = False
            else:
                keep_mask[j] = False
            cells_removed += 1

    print(f"    Removed {cells_removed} duplicate cells")
    print(f"    Remaining: {np.sum(keep_mask)} cells")

    return {
        'C': data['C'][:, keep_mask],
        'S': data['S'][:, keep_mask],
        'sn': data['sn'][keep_mask],
        'SNR': data['SNR'][keep_mask],
        'centroids': data['centroids'][keep_mask],
        'original_indices': data['original_indices'][keep_mask],
        'trial_times': data['trial_times'],
        'trial_direction': data['trial_direction'],
        'vr_time': data['vr_time'],
        'vr_position_x': data['vr_position_x'],
        'calcium_time': data['calcium_time'],
        'frame_rate': data['frame_rate'],
    }


def detect_active_cells_C_mode(
    C: np.ndarray,
    sn: np.ndarray,
    frame_rate: float,
    temporal_window_ms: float,
    amplitude_threshold_sigma: float,
) -> Tuple[np.ndarray, int]:
    """
    Detect active cells using C mode (denoised traces with 4sigma threshold).

    Args:
        C: Calcium traces (n_frames, n_neurons)
        sn: Noise levels per neuron
        frame_rate: Frame rate in Hz
        temporal_window_ms: Minimum separation between peaks (ms)
        amplitude_threshold_sigma: Amplitude threshold in sigma units

    Returns:
        Tuple of (active_mask, total_events)
    """
    n_frames, n_neurons = C.shape
    distance_frames = int(temporal_window_ms / 1000 * frame_rate)

    active_mask = np.zeros(n_neurons, dtype=bool)
    total_events = 0

    for i in range(n_neurons):
        trace = C[:, i]
        threshold = amplitude_threshold_sigma * sn[i]

        peaks, _ = find_peaks(trace, height=threshold, distance=distance_frames)

        if len(peaks) > 0:
            active_mask[i] = True
            total_events += len(peaks)

    return active_mask, total_events


def detect_active_cells_S_mode(
    S: np.ndarray,
    sn: np.ndarray,
    frame_rate: float,
) -> Tuple[np.ndarray, int]:
    """
    Detect active cells using S mode (deconvolved spikes).

    Since S data represents already deconvolved spike signals,
    we check for non-zero values rather than finding peaks.

    Args:
        S: Deconvolved spikes (n_frames, n_neurons)
        sn: Noise levels per neuron (unused but kept for API consistency)
        frame_rate: Frame rate in Hz (unused but kept for API consistency)

    Returns:
        Tuple of (active_mask, total_events)
    """
    n_frames, n_neurons = S.shape
    MIN_SPIKE_MAGNITUDE = 1e-6

    active_mask = np.zeros(n_neurons, dtype=bool)
    total_events = 0

    for i in range(n_neurons):
        trace = S[:, i]
        event_indices = np.where(trace > MIN_SPIKE_MAGNITUDE)[0]
        n_events = len(event_indices)

        if n_events > 0:
            active_mask[i] = True
            total_events += n_events

    return active_mask, total_events


def identify_global_active_cells(
    data: Dict[str, Any],
    mode: str,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Identify active cells globally across the entire session.

    Args:
        data: Preprocessed data dictionary
        mode: 'C' for denoised traces, 'S' for deconvolved spikes
        config: Configuration dictionary

    Returns:
        Tuple of (active_mask, active_cell_ids, total_events)
    """
    print(f"\n  Global active cell detection (Mode: {mode})")

    if mode == 'C':
        active_mask, total_events = detect_active_cells_C_mode(
            data['C'],
            data['sn'],
            data['frame_rate'],
            config['extraction']['temporal_window_ms'],
            config['extraction']['amplitude_threshold_sigma'],
        )
    else:  # mode == 'S'
        active_mask, total_events = detect_active_cells_S_mode(
            data['S'],
            data['sn'],
            data['frame_rate'],
        )

    n_active = np.sum(active_mask)
    active_cell_ids = data['original_indices'][active_mask]

    print(f"    Total events: {total_events}")
    print(f"    Active cells: {n_active} out of {len(active_mask)}")

    return active_mask, active_cell_ids, total_events


def extract_trial_data(
    data: Dict[str, Any],
    active_mask: np.ndarray,
    active_cell_ids: np.ndarray,
    mode: str,
    direction_map: Dict[int, str],
) -> list:
    """
    Extract trial-by-trial data using globally-defined active cells.

    Args:
        data: Preprocessed data dictionary
        active_mask: Boolean mask for active cells
        active_cell_ids: Original indices of active cells
        mode: 'C' or 'S'
        direction_map: Mapping from direction codes to 'L'/'R'

    Returns:
        List of trial data dictionaries
    """
    print(f"\n  Extracting trial data...")

    trial_times = data['trial_times']
    trial_directions = data['trial_direction']
    vr_time = data['vr_time']
    vr_position_x = data['vr_position_x']
    calcium_time = data['calcium_time']
    C = data['C']
    S = data['S']

    n_trials = len(trial_times)
    n_active_cells = np.sum(active_mask)

    trials = []

    for trial_idx in range(n_trials):
        start_time = trial_times[trial_idx]

        if trial_idx < n_trials - 1:
            end_time = trial_times[trial_idx + 1]
        else:
            end_time = calcium_time[-1]

        if start_time > calcium_time[-1] or end_time < calcium_time[0]:
            continue

        # Find calcium frames for this trial
        calcium_frame_mask = (calcium_time >= start_time) & (calcium_time < end_time)
        calcium_frame_indices = np.where(calcium_frame_mask)[0]

        if len(calcium_frame_indices) == 0:
            continue

        timestamps = calcium_time[calcium_frame_indices]
        position_x = np.interp(timestamps, vr_time, vr_position_x)

        # Extract neural activity for active cells only
        if mode == 'C':
            neural_activity = C[calcium_frame_indices][:, active_mask]
        else:
            neural_activity = S[calcium_frame_indices][:, active_mask]

        direction = direction_map.get(trial_directions[trial_idx], 'U')

        trials.append({
            'Trial': trial_idx,
            'Direction': direction,
            'start_time': start_time,
            'end_time': end_time,
            'TimeStamps': timestamps,
            'Position_x': position_x,
            'active_cell_ids': active_cell_ids,
            'active_cells_Activity': neural_activity,
            'n_active_cells': n_active_cells,
        })

    print(f"    Extracted {len(trials)} trials")
    print(f"    Active cells per trial: {n_active_cells}")

    return trials


def extract_active_cells(
    config: Dict[str, Any],
    input_mat_path: Optional[str] = None,
    mode: str = 'S',
) -> Path:
    """
    Main function to extract active cells and prepare downstream data.

    Args:
        config: Configuration dictionary
        input_mat_path: Path to merged MAT file (Pipeline D output)
        mode: 'C' for calcium traces, 'S' for deconvolved spikes

    Returns:
        Path to output MAT file
    """
    experiment_name = config['experiment']['name']

    print(f"\n{'='*60}")
    print(f"PIPELINE E: ACTIVE CELL EXTRACTION")
    print(f"{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Mode: {mode}")

    # Load input data
    if input_mat_path is None:
        # Try to find the most recent merged MAT file
        output_dir = Path(config['experiment']['output_dir'])
        mat_files = list(output_dir.glob(f"{experiment_name}*analyzed_data.mat"))
        if mat_files:
            input_mat_path = str(sorted(mat_files)[-1])
        else:
            raise FileNotFoundError(f"No merged MAT file found in {output_dir}")

    print(f"\nLoading: {input_mat_path}")
    mat_data = load_mat_file(input_mat_path)

    # Extract required data
    calcium = mat_data['calcium']
    trial = mat_data['trial']
    vr = mat_data['vr']
    sync = mat_data['sync']

    # Handle struct vs dict access
    def get_attr(obj, name):
        if hasattr(obj, name):
            return getattr(obj, name)
        return obj[name]

    raw_data = {
        'C': get_attr(calcium, 'C'),
        'S': get_attr(calcium, 'S'),
        'sn': get_attr(calcium, 'neurons_sn') if hasattr(calcium, 'neurons_sn') else np.ones(get_attr(calcium, 'C').shape[1]),
        'SNR': get_attr(calcium, 'SNR_comp') if hasattr(calcium, 'SNR_comp') else np.ones(get_attr(calcium, 'C').shape[1]),
        'centroids': get_attr(calcium, 'footprint_center') if hasattr(calcium, 'footprint_center') else np.zeros((get_attr(calcium, 'C').shape[1], 2)),
        'idx_good': get_attr(calcium, 'idx_components') if hasattr(calcium, 'idx_components') else np.arange(get_attr(calcium, 'C').shape[1]),
        'trial_times': get_attr(trial, 'timeSecs'),
        'trial_direction': get_attr(trial, 'direction'),
        'vr_time': get_attr(vr, 'timeSecs'),
        'vr_position_x': get_attr(vr, 'position_x'),
        'calcium_time': get_attr(sync, 'time_scope') if hasattr(sync, 'time_scope') else get_attr(sync, 'time_nidq'),
        'frame_rate': get_attr(calcium, 'frame_rate') if hasattr(calcium, 'frame_rate') else config['cnmfe']['frame_rate'],
    }

    print(f"  Frames: {raw_data['C'].shape[0]}")
    print(f"  Total neurons: {raw_data['C'].shape[1]}")

    # Step 1: Filter good cells
    filtered_data = filter_good_cells(raw_data)

    # Step 2: Remove duplicates
    clean_data = remove_duplicates_global(
        filtered_data,
        config['extraction']['neighbor_distance_um'],
        config['extraction']['pixel_size_um'],
        config['extraction']['correlation_threshold'],
    )

    # Step 3: Identify active cells globally
    active_mask, active_cell_ids, total_events = identify_global_active_cells(
        clean_data, mode, config
    )

    # Step 4: Extract trial data
    direction_map = config['extraction']['direction_map']
    trials = extract_trial_data(clean_data, active_mask, active_cell_ids, mode, direction_map)

    # Get sn values for active cells
    active_cells_sn = clean_data['sn'][active_mask]

    # Prepare output
    n_trials = len(trials)
    n_active_cells = len(active_cell_ids)

    Trial = np.array([t['Trial'] for t in trials], dtype=np.int32)
    Direction = np.array([t['Direction'] for t in trials], dtype='U1')
    start_time = np.array([t['start_time'] for t in trials], dtype=np.float64)
    end_time = np.array([t['end_time'] for t in trials], dtype=np.float64)

    TimeStamps = np.empty(n_trials, dtype=object)
    Position_x = np.empty(n_trials, dtype=object)
    active_cells_Activity = np.empty(n_trials, dtype=object)

    for i, t in enumerate(trials):
        TimeStamps[i] = t['TimeStamps'].astype(np.float64)
        Position_x[i] = t['Position_x'].astype(np.float64)
        active_cells_Activity[i] = t['active_cells_Activity'].astype(np.float64)

    output_data = {
        'Trial': Trial,
        'Direction': Direction,
        'start_time': start_time,
        'end_time': end_time,
        'TimeStamps': TimeStamps,
        'Position_x': Position_x,
        'active_cell_ids': active_cell_ids.astype(np.int32),
        'active_cells_sn': active_cells_sn.astype(np.float64),
        'active_cells_Activity': active_cells_Activity,
        'n_active_cells': n_active_cells,
        'parameters': {
            'mode': mode,
            'detection_scope': 'global',
            'frame_rate': config['cnmfe']['frame_rate'],
            'pixel_size_um': config['extraction']['pixel_size_um'],
            'neighbor_distance_um': config['extraction']['neighbor_distance_um'],
            'temporal_window_ms': config['extraction']['temporal_window_ms'],
            'amplitude_threshold_sigma': config['extraction']['amplitude_threshold_sigma'],
            'correlation_threshold': config['extraction']['correlation_threshold'],
        },
    }

    # Save output
    timestamp = datetime.now().strftime('%y%m%d')
    output_filename = f"{experiment_name}_{timestamp}An_downstream_{mode}.mat"
    output_path = get_output_path(config, output_filename)

    save_mat_file(str(output_path), output_data)

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"  Output: {output_path}")
    print(f"  Trials: {n_trials}")
    print(f"  Active cells: {n_active_cells}")
    print(f"  Mode: {mode}")
    print(f"{'='*60}")

    return output_path
