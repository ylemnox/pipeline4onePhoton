"""
I/O utility functions for 1p-spatial-pipeline.

Handles reading/writing MAT files, CSV files, NIDQ binary files, and metadata.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse
from scipy.io.matlab import mat_struct


def mat_struct_to_dict(obj: Any) -> Any:
    """
    Recursively convert mat_struct objects to Python dicts.

    Args:
        obj: Object to convert (mat_struct, ndarray, or other)

    Returns:
        Converted object with mat_structs replaced by dicts
    """
    if isinstance(obj, mat_struct):
        return {name: mat_struct_to_dict(getattr(obj, name)) for name in obj._fieldnames}
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        # Convert elements but keep as object array to handle inhomogeneous shapes
        converted = np.empty(obj.shape, dtype=object)
        for idx in np.ndindex(obj.shape):
            converted[idx] = mat_struct_to_dict(obj[idx])
        return converted
    else:
        return obj


def parse_nidq_meta(meta_path: str) -> Dict[str, str]:
    """
    Parse NIDQ .meta file.

    Args:
        meta_path: Path to .meta file

    Returns:
        Dictionary of metadata key-value pairs
    """
    meta = {}
    with open(meta_path, 'r') as f:
        for line in f:
            if '=' in line:
                k, v = line.split('=', 1)
                meta[k.strip()] = v.strip()
    return meta


def load_nidq_binary(bin_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load NIDQ binary file and metadata.

    Args:
        bin_path: Path to .nidq.bin file

    Returns:
        Tuple of (digital_channel_data, metadata_dict)
    """
    bin_path = Path(bin_path)
    meta_path = bin_path.with_suffix('.meta')

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta = parse_nidq_meta(str(meta_path))
    n_chans = int(meta['nSavedChans'])
    fs_nidq = float(meta['niSampRate'])

    file_size = bin_path.stat().st_size
    n_samples = file_size // (2 * n_chans)

    raw_data = np.memmap(bin_path, dtype='int16', mode='r', shape=(n_samples, n_chans))
    digital_ch = raw_data[:, 0]

    return digital_ch, {
        'n_channels': n_chans,
        'sample_rate': fs_nidq,
        'n_samples': n_samples,
        'duration_seconds': n_samples / fs_nidq,
    }


def extract_nidq_events(digital_channel: np.ndarray, sample_rate: float) -> Dict[str, np.ndarray]:
    """
    Extract events from all 8 bits of the NIDQ digital channel.

    Args:
        digital_channel: Digital channel data (int16)
        sample_rate: NIDQ sample rate in Hz

    Returns:
        Dictionary with time, frame, chan, type arrays for all events
    """
    all_events = []

    for bit in range(8):
        sig = (digital_channel >> bit) & 1
        diff = np.diff(sig.astype(np.int8))

        rising = np.where(diff == 1)[0] + 1
        falling = np.where(diff == -1)[0] + 1

        for f in rising:
            all_events.append({
                'time': f / sample_rate,
                'frame': f,
                'chan': bit + 1,
                'type': 1  # rising
            })
        for f in falling:
            all_events.append({
                'time': f / sample_rate,
                'frame': f,
                'chan': bit + 1,
                'type': 0  # falling
            })

    if not all_events:
        return {
            'time': np.array([]),
            'frame': np.array([]),
            'chan': np.array([]),
            'type': np.array([]),
        }

    # Sort by time
    time_all = np.array([e['time'] for e in all_events])
    sort_idx = np.argsort(time_all)

    return {
        'time': time_all[sort_idx],
        'frame': np.array([e['frame'] for e in all_events])[sort_idx],
        'chan': np.array([e['chan'] for e in all_events])[sort_idx],
        'type': np.array([e['type'] for e in all_events])[sort_idx],
    }


def load_deeplabcut_csv(csv_path: str, likelihood_threshold: float = 0.9) -> pd.DataFrame:
    """
    Load DeepLabCut output CSV.

    Args:
        csv_path: Path to DLC CSV file
        likelihood_threshold: Filter threshold for predictions

    Returns:
        DataFrame with frame, x, y, likelihood columns
    """
    df = pd.read_csv(csv_path, skiprows=3, header=None)
    df.columns = ['frame', 'x', 'y', 'likelihood']

    # Interpolate low-confidence points
    low_likelihood = df['likelihood'] < likelihood_threshold
    if low_likelihood.any():
        df.loc[low_likelihood, 'x'] = np.nan
        df.loc[low_likelihood, 'y'] = np.nan
        df['x'] = df['x'].interpolate(method='linear', limit_direction='both')
        df['y'] = df['y'].interpolate(method='linear', limit_direction='both')

    return df


def load_metadata_json(json_path: str) -> Dict[str, Any]:
    """
    Load Miniscope DAQ metaData.json.

    Args:
        json_path: Path to metaData.json

    Returns:
        Metadata dictionary
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def load_mat_file(mat_path: str) -> Dict[str, Any]:
    """
    Load MAT file with struct handling.

    Args:
        mat_path: Path to MAT file

    Returns:
        Dictionary of MAT file contents
    """
    return sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)


def save_mat_file(mat_path: str, data: Dict[str, Any]) -> None:
    """
    Save data to MAT file.

    Args:
        mat_path: Output path
        data: Dictionary to save
    """
    sio.savemat(mat_path, data, oned_as='column')


def process_sparse_matrix(name: str, sparse_mat) -> Dict[str, np.ndarray]:
    """
    Convert sparse matrix to components for MAT saving.

    Args:
        name: Name prefix for the components
        sparse_mat: Sparse matrix (will be converted to CSC)

    Returns:
        Dictionary with data, indices, indptr, shape arrays
    """
    if not scipy.sparse.issparse(sparse_mat):
        return {name: sparse_mat}

    if not isinstance(sparse_mat, scipy.sparse.csc_matrix):
        sparse_mat = sparse_mat.tocsc()

    return {
        f'{name}_sparse_data': sparse_mat.data,
        f'{name}_sparse_indices': sparse_mat.indices,
        f'{name}_sparse_indptr': sparse_mat.indptr,
        f'{name}_sparse_shape': np.array(sparse_mat.shape, dtype=np.int32),
    }


def convert_pixels_to_cm(
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
    track_length_cm: float,
    flip_x: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel coordinates to cm.

    Args:
        x_pixels: X coordinates in pixels
        y_pixels: Y coordinates in pixels
        track_length_cm: Track length in cm
        flip_x: If True, flip x-axis (smallest x -> right end, largest x -> left end)

    Returns:
        Tuple of (x_cm, y_cm)
    """
    x_min, x_max = x_pixels.min(), x_pixels.max()
    y_min, y_max = y_pixels.min(), y_pixels.max()
    x_range_pixels = x_max - x_min

    pixels_per_cm = x_range_pixels / track_length_cm

    if flip_x:
        position_x = track_length_cm - ((x_pixels - x_min) / pixels_per_cm)
    else:
        position_x = (x_pixels - x_min) / pixels_per_cm

    position_y = (y_pixels - y_min) / pixels_per_cm

    return position_x, position_y
