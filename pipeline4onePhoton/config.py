"""
Configuration handling for 1p-spatial-pipeline.

Loads and validates YAML configuration files with sensible defaults.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


# Default configuration values
DEFAULTS = {
    'experiment': {
        'name': 'experiment',
        'output_dir': './output',
    },
    'behavior': {
        'position_csv': None,  # Required
        'likelihood_threshold': 0.9,
        'track_length_cm': 48,
        'flip_direction': False,
        'webcam_fps': 32.0,
    },
    'cnmfe': {
        'mat_file': None,  # Path to pre-computed CNMF-E MAT file
        'video_dir': None,  # Required for Pipeline B (if running CNMF-E)
        'frame_rate': 20,
        'gSig': [4, 4],
        'gSiz': [13, 13],
        'min_corr': 0.8,
        'min_pnr': 10,
        'K': None,  # Auto-detect
        'Ain': None,
        'ssub': 2,
        'tsub': 2,
        'p': 1,
        'merge_thr': 0.85,
        'rf': 40,
        'stride': 20,
        'only_init': True,
        'nb': 0,
        'nb_patch': 0,
        'method_deconvolution': 'oasis',
        'low_rank_background': None,
        'update_background_components': True,
        'min_SNR': 3,
        'rval_thr': 0.85,
        'use_cnn': True,
        'min_cnn_thr': 0.99,
        'cnn_lowest': 0.1,
    },
    'nidq': {
        'bin_file': None,  # Required for Pipeline C
        'meta_file': None,  # Required for Pipeline C
        'sync_bit': 2,
        'trigger_bit': 1,
        'reward_left_bit': 5,
        'reward_right_bit': 4,
    },
    'metadata': {
        'json_file': None,  # Optional metaData.json path
    },
    'extraction': {
        'pixel_size_um': 1.67,
        'neighbor_distance_um': 15,
        'correlation_threshold': 0.9,
        'temporal_window_ms': 300,
        'amplitude_threshold_sigma': 4,
        'direction_map': {0: 'R', 1: 'L'},
    },
    'analysis': {
        'n_bins': 48,
        'smoothing_sigma_cm': 4.5,
        'min_speed_cm_s': 0.5,
        'reward_zone_cm': 1.5,
        'mi_bin_size_cm': 4,
        'n_shuffles': 10000,
        'p_threshold': 0.05,
    },
}


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge override dict into base dict.

    Override values take precedence. Nested dicts are merged, not replaced.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Merges user config with defaults. Missing values use defaults.

    Args:
        config_path: Path to YAML config file. If None, returns defaults.

    Returns:
        Configuration dictionary

    Raises:
        ConfigError: If config file is invalid or missing required fields
    """
    config = DEFAULTS.copy()

    if config_path is None:
        return config

    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in config file: {e}")

    if user_config is None:
        return config

    # Deep merge user config with defaults
    config = deep_merge(DEFAULTS, user_config)

    return config


def validate_config(config: Dict[str, Any], pipelines: list) -> Dict[str, Any]:
    """
    Validate configuration for specified pipelines.

    Args:
        config: Configuration dictionary
        pipelines: List of pipeline letters to validate (e.g., ['C', 'D', 'E'])

    Returns:
        Validated configuration

    Raises:
        ConfigError: If required fields are missing or invalid
    """
    errors = []

    # Common validation
    if not config.get('experiment', {}).get('name'):
        errors.append("experiment.name is required")

    # Pipeline-specific validation
    if 'C' in pipelines:
        if not config.get('nidq', {}).get('bin_file'):
            errors.append("nidq.bin_file is required for Pipeline C")
        if not config.get('nidq', {}).get('meta_file'):
            errors.append("nidq.meta_file is required for Pipeline C")

    if 'D' in pipelines:
        if not config.get('behavior', {}).get('position_csv'):
            errors.append("behavior.position_csv is required for Pipeline D")
        if not config.get('nidq', {}).get('bin_file'):
            errors.append("nidq.bin_file is required for Pipeline D")

    if errors:
        raise ConfigError("Configuration validation failed:\n  - " + "\n  - ".join(errors))

    return config


def get_output_path(config: Dict[str, Any], filename: str, subdir: Optional[str] = None) -> Path:
    """
    Get output file path based on config.

    Args:
        config: Configuration dictionary
        filename: Output filename
        subdir: Optional subdirectory

    Returns:
        Full output path
    """
    output_dir = Path(config['experiment']['output_dir'])
    experiment_name = config['experiment']['name']

    if subdir:
        output_dir = output_dir / subdir

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / filename


def generate_config_template(
    experiment_name: str = "MY01_20251231",
    output_path: Optional[str] = None
) -> str:
    """
    Generate a configuration template with documented options.

    Args:
        experiment_name: Name for the experiment
        output_path: Optional path to write the config file

    Returns:
        Configuration template as string
    """
    template = f'''# 1p-spatial-pipeline Configuration
# Generated for experiment: {experiment_name}
# See documentation at: https://github.com/ylemnox/1p-spatial-pipeline

# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================
experiment:
  name: "{experiment_name}"
  output_dir: "./output/{experiment_name}"

# =============================================================================
# PIPELINE A OUTPUT (DeepLabCut - processed externally)
# =============================================================================
# DeepLabCut must be run separately. Provide the output CSV here.
behavior:
  # Path to DeepLabCut output CSV (frame, x, y, likelihood)
  position_csv: "./data/deepLabCut_result_{experiment_name}.csv"

  # Filter threshold for DLC predictions (0-1)
  likelihood_threshold: 0.9

  # Linear track length in cm
  track_length_cm: 48

  # Set true if left/right directions are reversed in your setup
  flip_direction: false

  # Webcam frame rate for temporal alignment
  webcam_fps: 32.0

# =============================================================================
# PIPELINE B: CNMF-E (CaImAn) - Run Externally
# =============================================================================
# Pipeline B must be run externally via notebooks/cnmfE_pipeline.ipynb
# (Google Colab recommended). Provide the output MAT file path below.
#
# See docs/pipeline_b_caiman_setup.md for detailed instructions.
cnmfe:
  # Path to pre-computed CNMF-E MAT file (from notebooks/cnmfE_pipeline.ipynb)
  mat_file: "./data/cnmfe_output_{experiment_name}.mat"

  # Miniscope frame rate in Hz
  frame_rate: 20

  # --- CNMF-E Parameters (for notebook reference) ---
  # Gaussian kernel size for cell detection (half-width in pixels)
  gSig: [4, 4]

  # Average neuron size (diameter in pixels)
  gSiz: [13, 13]

  # Minimum local correlation for cell detection
  min_corr: 0.8

  # Minimum peak-to-noise ratio
  min_pnr: 10

# =============================================================================
# PIPELINE C: NIDQ Signals
# =============================================================================
nidq:
  # Path to NIDQ binary file
  bin_file: "./data/20251231_{experiment_name}_g0_t0.nidq.bin"

  # Path to NIDQ metadata file
  meta_file: "./data/20251231_{experiment_name}_g0_t0.nidq.meta"

  # --- Channel Assignments (determine via '1p-spatial-pipeline nidq' command) ---
  # Bit number for miniscope sync signal
  sync_bit: 2

  # Bit number for webcam trigger signal
  trigger_bit: 1

  # Bit number for left reward signal
  reward_left_bit: 5

  # Bit number for right reward signal
  reward_right_bit: 4

# =============================================================================
# METADATA (Optional)
# =============================================================================
metadata:
  # Path to Miniscope DAQ metaData.json
  json_file: "./miniscope/{experiment_name}/metaData.json"

# =============================================================================
# PIPELINE E: Active Cell Extraction
# =============================================================================
extraction:
  # Pixel size for spatial calculations (um/pixel)
  pixel_size_um: 1.67

  # Distance threshold for neighbor detection (um)
  neighbor_distance_um: 15

  # Correlation threshold for duplicate cell removal
  correlation_threshold: 0.9

  # --- C Mode Parameters (denoised calcium traces) ---
  # Minimum separation between events (ms)
  temporal_window_ms: 300

  # Amplitude threshold (multiples of noise level)
  amplitude_threshold_sigma: 4

  # Direction mapping (from trial data to L/R)
  direction_map:
    0: "R"
    1: "L"

# =============================================================================
# PIPELINE F: Place Field Analysis
# =============================================================================
analysis:
  # Number of spatial bins (track_length / n_bins = bin size in cm)
  n_bins: 48

  # Gaussian smoothing sigma (cm)
  smoothing_sigma_cm: 4.5

  # Minimum speed for valid movement (cm/s)
  min_speed_cm_s: 0.5

  # Reward zone exclusion at each end (cm)
  reward_zone_cm: 1.5

  # Bin size for mutual information calculation (cm)
  mi_bin_size_cm: 4

  # Number of shuffle iterations for significance testing
  n_shuffles: 10000

  # P-value threshold for place cell identification
  p_threshold: 0.05
'''

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(template)

    return template


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print a summary of the current configuration."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(title="Configuration Summary", show_header=True)
    table.add_column("Section", style="cyan")
    table.add_column("Parameter", style="green")
    table.add_column("Value", style="yellow")

    # Experiment
    table.add_row("experiment", "name", str(config['experiment']['name']))
    table.add_row("experiment", "output_dir", str(config['experiment']['output_dir']))

    # Behavior
    table.add_row("behavior", "position_csv", str(config['behavior'].get('position_csv', 'Not set')))
    table.add_row("behavior", "track_length_cm", str(config['behavior']['track_length_cm']))

    # CNMF-E
    table.add_row("cnmfe", "mat_file", str(config['cnmfe'].get('mat_file', 'Not set')))

    # NIDQ
    table.add_row("nidq", "bin_file", str(config['nidq'].get('bin_file', 'Not set')))
    table.add_row("nidq", "sync_bit", str(config['nidq']['sync_bit']))

    # Extraction
    table.add_row("extraction", "correlation_threshold", str(config['extraction']['correlation_threshold']))

    # Analysis
    table.add_row("analysis", "n_shuffles", str(config['analysis']['n_shuffles']))
    table.add_row("analysis", "p_threshold", str(config['analysis']['p_threshold']))

    console.print(table)
