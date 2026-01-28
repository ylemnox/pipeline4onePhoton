"""
Pipeline D: Merge data from Pipelines A, B, and C.

Combines:
- Position data from DeepLabCut CSV (Pipeline A)
- Calcium data from CNMF-E MAT file (Pipeline B)
- Synchronization signals from NIDQ (Pipeline C)

Output: Merged MAT file ready for downstream analysis.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ..config import get_output_path
from ..utils.io_utils import (
    load_mat_file,
    save_mat_file,
    load_deeplabcut_csv,
    load_metadata_json,
    convert_pixels_to_cm,
    mat_struct_to_dict,
)
from ..pipeline_c.nidq_processor import process_nidq_signals, extract_reward_events


def merge_all_data(
    config: Dict[str, Any],
    cnmfe_mat_path: Optional[str] = None,
) -> Path:
    """
    Merge all data sources into a single MAT file.

    Args:
        config: Configuration dictionary
        cnmfe_mat_path: Optional path to pre-computed CNMF-E MAT file

    Returns:
        Path to output MAT file
    """
    experiment_name = config['experiment']['name']
    frame_rate = config['cnmfe']['frame_rate']

    print(f"\n{'='*60}")
    print("PIPELINE D: DATA MERGE")
    print(f"{'='*60}")
    print(f"Experiment: {experiment_name}")

    mat_data = {}

    # =========================================================================
    # Load CNMF-E data (Pipeline B output)
    # =========================================================================
    if cnmfe_mat_path:
        print(f"\n[1] Loading CNMF-E data from: {cnmfe_mat_path}")
        cnmfe_data = load_mat_file(cnmfe_mat_path)

        if 'calcium' in cnmfe_data:
            mat_data['calcium'] = mat_struct_to_dict(cnmfe_data['calcium'])
            print(f"    Loaded calcium structure")
        else:
            # Create calcium structure from top-level fields
            mat_data['calcium'] = {}
            for key in ['C', 'S', 'F_dff', 'A', 'b', 'f', 'Cn', 'idx_components']:
                if key in cnmfe_data:
                    mat_data['calcium'][key] = cnmfe_data[key]
                    print(f"    Loaded {key}")

        # Get frame count from C matrix
        if hasattr(mat_data.get('calcium', {}), 'C'):
            n_frames = mat_data['calcium'].C.shape[0]
        elif 'C' in mat_data.get('calcium', {}):
            n_frames = mat_data['calcium']['C'].shape[0]
        else:
            n_frames = 0
            print("    WARNING: Could not determine frame count from CNMF-E data")

        mat_data['calcium']['frame_rate'] = float(frame_rate)
    else:
        print("\n[1] No CNMF-E MAT file provided - skipping calcium data")
        n_frames = 0

    # =========================================================================
    # Process NIDQ signals (Pipeline C)
    # =========================================================================
    print(f"\n[2] Processing NIDQ signals...")
    try:
        sync_struct, nidq_struct = process_nidq_signals(config)

        # Create miniscope time axis
        sync_start = sync_struct['sync_start_nidq']
        sync_end = sync_struct['sync_end_nidq']

        if n_frames == 0:
            # Estimate frame count from sync duration
            n_frames = int((sync_end - sync_start) * frame_rate)
            print(f"    Estimated frames from sync: {n_frames}")

        frame_indices = np.arange(n_frames, dtype=np.float64)
        scope_times = sync_start + (frame_indices / frame_rate)

        sync_struct['time_scope'] = scope_times
        sync_struct['frame_scope'] = frame_indices

        mat_data['sync'] = sync_struct
        mat_data['nidq'] = nidq_struct

        print(f"    Sync: {sync_start:.2f}s - {sync_end:.2f}s")
        print(f"    Behavioral events: {len(nidq_struct['time'])}")

    except Exception as e:
        print(f"    WARNING: NIDQ processing failed: {e}")
        print("    Creating placeholder sync structure...")

        # Create basic time axis without NIDQ
        frame_indices = np.arange(n_frames, dtype=np.float64)
        scope_times = frame_indices / frame_rate

        mat_data['sync'] = {
            'time_scope': scope_times,
            'frame_scope': frame_indices,
        }

    # =========================================================================
    # Load and align position data (Pipeline A output)
    # =========================================================================
    print(f"\n[3] Loading position data...")
    position_csv = config['behavior'].get('position_csv')

    if position_csv and Path(position_csv).exists():
        likelihood_threshold = config['behavior']['likelihood_threshold']
        track_length_cm = config['behavior']['track_length_cm']
        webcam_fps = config['behavior']['webcam_fps']
        flip_direction = config['behavior']['flip_direction']

        df_pos = load_deeplabcut_csv(position_csv, likelihood_threshold)
        print(f"    Loaded {len(df_pos)} webcam frames")

        # Convert to cm
        position_x, position_y = convert_pixels_to_cm(
            df_pos['x'].values,
            df_pos['y'].values,
            track_length_cm,
            flip_x=not flip_direction,
        )
        print(f"    Position range: X=[{position_x.min():.1f}, {position_x.max():.1f}] cm")

        # Create webcam time axis using trigger channel
        trigger_bit = config['nidq']['trigger_bit']
        trigger_chan = trigger_bit + 1

        if 'nidq' in mat_data:
            trigger_mask = mat_data['nidq']['chan'] == trigger_chan
            trigger_times = mat_data['nidq']['time'][trigger_mask]
            trigger_types = mat_data['nidq']['type'][trigger_mask]

            falling_mask = trigger_types == 0
            if falling_mask.any():
                trigger_falling = trigger_times[falling_mask]
                trigger_end = trigger_falling[-1]

                n_webcam_frames = len(df_pos)
                webcam_time_axis = trigger_end - ((n_webcam_frames - 1 - np.arange(n_webcam_frames)) / webcam_fps)
                print(f"    Webcam aligned using trigger: {webcam_time_axis[0]:.2f}s - {webcam_time_axis[-1]:.2f}s")
            else:
                # Fallback to sync
                webcam_time_axis = mat_data['sync']['time_scope'][0] + (np.arange(len(df_pos)) / webcam_fps)
                print(f"    Webcam aligned using sync (fallback)")
        else:
            webcam_time_axis = np.arange(len(df_pos)) / webcam_fps
            print(f"    Webcam time axis (no NIDQ)")

        # Interpolate to miniscope frame times
        miniscope_time_axis = mat_data['sync']['time_scope']

        f_x = interp1d(webcam_time_axis, position_x, kind='linear',
                       bounds_error=False, fill_value=np.nan)
        f_y = interp1d(webcam_time_axis, position_y, kind='linear',
                       bounds_error=False, fill_value=np.nan)

        position_x_aligned = f_x(miniscope_time_axis)
        position_y_aligned = f_y(miniscope_time_axis)

        # Calculate speed
        position_2d = np.column_stack([position_x_aligned, position_y_aligned])
        dt = np.mean(np.diff(miniscope_time_axis)) if len(miniscope_time_axis) > 1 else 1.0/frame_rate
        dx = np.sqrt(np.sum(np.diff(position_2d, axis=0)**2, axis=1))
        speed = np.concatenate([[np.nan], dx / dt])

        # Smooth speed
        speed_smooth = pd.Series(speed).rolling(5, center=True, min_periods=1).mean().values

        # Build VR structure
        mat_data['vr'] = {
            'timeSecs': miniscope_time_axis,
            'frame': mat_data['sync']['frame_scope'],
            'position_x': position_x_aligned,
            'position_y': position_y_aligned,
            'position_z': np.zeros(n_frames),
            'speed': speed_smooth,
            'rotation': np.zeros(n_frames),
            'distance': np.concatenate([[0], np.cumsum(np.abs(np.diff(position_x_aligned)))]),
        }

        valid_mask = ~np.isnan(position_x_aligned)
        print(f"    Interpolated: {valid_mask.sum()}/{len(valid_mask)} valid frames")

    else:
        print(f"    WARNING: Position CSV not found: {position_csv}")

    # =========================================================================
    # Extract trial structure from reward events
    # =========================================================================
    print(f"\n[4] Extracting trial structure...")

    if 'nidq' in mat_data and 'vr' in mat_data:
        left_rewards, right_rewards = extract_reward_events(mat_data['nidq'], config)
        print(f"    Left rewards: {len(left_rewards)}")
        print(f"    Right rewards: {len(right_rewards)}")

        # Combine and sort rewards
        all_rewards = []
        for t in left_rewards:
            all_rewards.append({'time': t, 'side': 1})  # 1 = left
        for t in right_rewards:
            all_rewards.append({'time': t, 'side': 0})  # 0 = right

        all_rewards.sort(key=lambda x: x['time'])
        n_trials = len(all_rewards)

        if n_trials > 0:
            trial_times = np.array([r['time'] for r in all_rewards])
            trial_sides = np.array([r['side'] for r in all_rewards])

            # Get position/speed at reward
            vr_times = mat_data['vr']['timeSecs']
            vr_position_x = mat_data['vr']['position_x']
            vr_speed = mat_data['vr']['speed']

            trial_positions = np.zeros(n_trials)
            trial_speeds = np.zeros(n_trials)

            for i, reward_time in enumerate(trial_times):
                closest_idx = np.argmin(np.abs(vr_times - reward_time))
                trial_positions[i] = vr_position_x[closest_idx]
                trial_speeds[i] = vr_speed[closest_idx]

            # Determine direction
            trial_directions = np.zeros(n_trials)
            for i in range(n_trials):
                if i == 0:
                    trial_directions[i] = 1 if trial_sides[i] == 1 else 0
                else:
                    trial_directions[i] = 1 if trial_positions[i] > trial_positions[i-1] else 0

            # Inter-trial intervals
            iti = np.concatenate([[0], np.diff(trial_times)])

            mat_data['trial'] = {
                'timeSecs': trial_times,
                'iTrial': np.arange(n_trials),
                'choice': trial_sides,
                'result': np.ones(n_trials),
                'reward': np.ones(n_trials),
                'position_at_reward': trial_positions,
                'speed_at_reward': trial_speeds,
                'direction': trial_directions,
                'iti': iti,
                'n_trial': int(n_trials),
            }

            print(f"    Extracted {n_trials} trials")
            print(f"    Left: {(trial_sides == 1).sum()}, Right: {(trial_sides == 0).sum()}")
        else:
            print("    No reward events found")
    else:
        print("    Skipping trial extraction (missing NIDQ or VR data)")

    # =========================================================================
    # Load metadata
    # =========================================================================
    print(f"\n[5] Loading metadata...")
    metadata_path = config['metadata'].get('json_file')

    if metadata_path and Path(metadata_path).exists():
        metadata = load_metadata_json(metadata_path)

        mat_data['task_info'] = {
            'animalName': metadata.get('animalName', 'Unknown'),
            'task': metadata.get('experimentName', 'Unknown'),
            'researcherName': metadata.get('researcherName', 'Unknown'),
            'nTrial': mat_data.get('trial', {}).get('n_trial', 0),
        }

        if 'recordingStartTime' in metadata:
            rst = metadata['recordingStartTime']
            mat_data['task_info'].update({
                'year': rst.get('year', 0),
                'month': rst.get('month', 0),
                'day': rst.get('day', 0),
                'hour': rst.get('hour', 0),
                'minute': rst.get('minute', 0),
                'second': rst.get('second', 0),
            })
            mat_data['task_time'] = rst.get('msecSinceEpoch', 0) / 1000.0

        mat_data['task_parameter'] = {
            'baseDirectory': metadata.get('baseDirectory', ''),
            'miniscope': metadata.get('miniscopes', [''])[0] if metadata.get('miniscopes') else '',
            'camera': metadata.get('cameras', [''])[0] if metadata.get('cameras') else '',
        }

        mat_data['task_path'] = str(Path(metadata_path).absolute())
        mat_data['task_type'] = 'linear_track'

        print(f"    Animal: {mat_data['task_info']['animalName']}")
        print(f"    Task: {mat_data['task_info']['task']}")
    else:
        print(f"    Metadata not found: {metadata_path}")
        mat_data['task_info'] = {'animalName': experiment_name}

    # =========================================================================
    # Save merged MAT file
    # =========================================================================
    print(f"\n[6] Saving merged MAT file...")

    timestamp = datetime.now().strftime('%y%m%d')
    output_filename = f"{experiment_name}_{timestamp}analyzed_data.mat"
    output_path = get_output_path(config, output_filename)

    save_mat_file(str(output_path), mat_data)

    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"\n{'='*60}")
    print(f"Merge complete!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Structures: {[k for k in mat_data.keys() if not k.startswith('_')]}")
    print(f"{'='*60}")

    return output_path
