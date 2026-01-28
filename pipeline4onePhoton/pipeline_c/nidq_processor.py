"""
Pipeline C: NIDQ signal processing.

Visualizes and processes NIDQ digital signals to identify:
- Sync channel (miniscope frame sync)
- Trigger channel (webcam trigger)
- Reward channels (left/right)
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

from ..utils.io_utils import load_nidq_binary, extract_nidq_events, parse_nidq_meta

console = Console()


def analyze_channel_profile(events: Dict[str, np.ndarray], channel: int) -> Dict[str, Any]:
    """
    Analyze the profile of a single NIDQ channel.

    Args:
        events: All events dictionary
        channel: Channel number (1-8)

    Returns:
        Profile dictionary with statistics
    """
    mask = events['chan'] == channel
    channel_times = events['time'][mask]
    channel_types = events['type'][mask]

    if len(channel_times) == 0:
        return {
            'channel': channel,
            'n_events': 0,
            'n_rising': 0,
            'n_falling': 0,
            'first_event': None,
            'last_event': None,
            'duration': 0,
            'mean_frequency': 0,
            'description': 'No events',
        }

    rising_times = channel_times[channel_types == 1]
    falling_times = channel_times[channel_types == 0]

    # Calculate frequency from rising edges
    if len(rising_times) > 1:
        intervals = np.diff(rising_times)
        mean_freq = 1.0 / np.mean(intervals) if np.mean(intervals) > 0 else 0
    else:
        mean_freq = 0

    first_event = channel_times.min()
    last_event = channel_times.max()
    duration = last_event - first_event

    # Guess channel type based on frequency
    if 9 < mean_freq < 11:
        description = "~10 Hz - likely SYNC (miniscope)"
    elif len(rising_times)==1 & len(falling_times)==1:
        description = "likely TRIGGER (webcam)"
    elif mean_freq < 1.5:
        description = "< 1.5 Hz - likely REWARD or SPARSE"
    else:
        description = f"~{mean_freq:.1f} Hz"

    return {
        'channel': channel,
        'n_events': len(channel_times),
        'n_rising': len(rising_times),
        'n_falling': len(falling_times),
        'first_event': first_event,
        'last_event': last_event,
        'duration': duration,
        'mean_frequency': mean_freq,
        'description': description,
    }


def visualize_nidq_channels(config: Dict[str, Any], preview_seconds: float = 10.0) -> None:
    """
    Visualize NIDQ channels and display profiles.

    Args:
        config: Configuration dictionary
        preview_seconds: Duration to preview in seconds
    """
    bin_path = config['nidq']['bin_file']

    console.print(f"\n[cyan]Loading NIDQ file:[/cyan] {bin_path}")

    # Load NIDQ data
    digital_ch, meta = load_nidq_binary(bin_path)
    console.print(f"  Sample rate: {meta['sample_rate']:.2f} Hz")
    console.print(f"  Duration: {meta['duration_seconds']:.2f} seconds")
    console.print(f"  Samples: {meta['n_samples']:,}")

    # Extract events
    console.print("\n[cyan]Extracting events from all channels...[/cyan]")
    events = extract_nidq_events(digital_ch, meta['sample_rate'])
    console.print(f"  Total events: {len(events['time']):,}")

    # Analyze each channel
    console.print("\n[bold]Channel Profiles:[/bold]")

    table = Table(show_header=True)
    table.add_column("Channel", style="cyan", justify="center")
    table.add_column("Events", justify="right")
    table.add_column("Rising", justify="right")
    table.add_column("Falling", justify="right")
    table.add_column("Start (s)", justify="right")
    table.add_column("End (s)", justify="right")
    table.add_column("Freq (Hz)", justify="right")
    table.add_column("Description", style="yellow")

    profiles = []
    for ch in range(1, 9):
        profile = analyze_channel_profile(events, ch)
        profiles.append(profile)

        if profile['n_events'] > 0:
            table.add_row(
                str(ch),
                str(profile['n_events']),
                str(profile['n_rising']),
                str(profile['n_falling']),
                f"{profile['first_event']:.2f}" if profile['first_event'] else "-",
                f"{profile['last_event']:.2f}" if profile['last_event'] else "-",
                f"{profile['mean_frequency']:.2f}" if profile['mean_frequency'] else "-",
                profile['description'],
            )
        else:
            table.add_row(str(ch), "0", "0", "0", "-", "-", "-", "No events")

    console.print(table)

    # Create visualization
    console.print(f"\n[cyan]Generating signal preview ({preview_seconds}s)...[/cyan]")

    preview_samples = int(preview_seconds * meta['sample_rate'])
    preview_samples = min(preview_samples, len(digital_ch))

    fig, axes = plt.subplots(8, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'NIDQ Digital Channels (first {preview_seconds}s)', fontsize=14)

    time_axis = np.arange(preview_samples) / meta['sample_rate']

    for bit in range(8):
        ax = axes[bit]
        channel = bit + 1
        sig = (digital_ch[:preview_samples] >> bit) & 1

        ax.plot(time_axis, sig, 'b-', linewidth=0.5)
        ax.set_ylabel(f'Ch {channel}', fontsize=10)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])

        # Add profile info
        profile = profiles[bit]
        if profile['n_events'] > 0:
            ax.text(0.98, 0.85, profile['description'],
                   transform=ax.transAxes, fontsize=9,
                   ha='right', va='top', color='red')

    axes[-1].set_xlabel('Time (seconds)')

    plt.tight_layout()

    # Save figure
    output_dir = Path(config['experiment'].get('output_dir', '.'))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / 'nidq_channel_preview.png'
    plt.show()
    #plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    #plt.close()

    console.print(f"\n[green]Preview saved to:[/green] {fig_path}")

    # Show recommendations
    console.print("\n[bold]Channel Assignment Recommendations:[/bold]")

    sync_candidates = [p for p in profiles if 9 < p['mean_frequency'] < 11]
    trigger_candidates = [p for p in profiles if 30 < p['mean_frequency'] < 35]
    reward_candidates = [p for p in profiles if 0 < p['mean_frequency'] < 1]

    if sync_candidates:
        ch = sync_candidates[0]['channel']
        console.print(f"  [cyan]sync_bit:[/cyan] {ch - 1} (channel {ch}, ~10 Hz)")
    else:
        console.print("  [yellow]sync_bit:[/yellow] Not auto-detected")

    if trigger_candidates:
        ch = trigger_candidates[0]['channel']
        console.print(f"  [cyan]trigger_bit:[/cyan] {ch - 1} (channel {ch}, ~32 Hz)")
    else:
        console.print("  [yellow]trigger_bit:[/yellow] Not auto-detected")

    if len(reward_candidates) >= 2:
        console.print(f"  [cyan]reward candidates:[/cyan] channels {[p['channel'] for p in reward_candidates]}")
    elif reward_candidates:
        console.print(f"  [cyan]reward candidate:[/cyan] channel {reward_candidates[0]['channel']}")

    console.print("\n[dim]Update your config.yaml with the correct channel assignments.[/dim]")


def process_nidq_signals(
    config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process NIDQ signals and separate sync from behavioral events.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (sync_structure, nidq_structure) for MAT file
    """
    bin_path = config['nidq']['bin_file']
    sync_bit = config['nidq']['sync_bit']
    sync_chan = sync_bit + 1

    # Load and extract events
    digital_ch, meta = load_nidq_binary(bin_path)
    events = extract_nidq_events(digital_ch, meta['sample_rate'])

    if len(events['time']) == 0:
        raise ValueError("No NIDQ events found")

    # Separate sync from behavioral events
    sync_mask = events['chan'] == sync_chan
    behav_mask = ~sync_mask

    sync_time = events['time'][sync_mask]
    sync_frame = events['frame'][sync_mask]
    sync_type = events['type'][sync_mask]

    # Get sync start/end (using falling edges)
    falling_mask = sync_type == 0
    if falling_mask.any():
        sync_start = sync_time[falling_mask][0]
        sync_end = sync_time[falling_mask][-1]
    else:
        sync_start = sync_time[0] if len(sync_time) > 0 else 0
        sync_end = sync_time[-1] if len(sync_time) > 0 else 0

    # Build sync structure
    sync_struct = {
        'time_nidq': sync_time,
        'frame_nidq': sync_frame,
        'type_nidq': sync_type,
        'sync_start_nidq': sync_start,
        'sync_end_nidq': sync_end,
        'sample_rate': meta['sample_rate'],
    }

    # Build nidq structure (behavioral only)
    nidq_struct = {
        'time': events['time'][behav_mask],
        'frame': events['frame'][behav_mask],
        'chan': events['chan'][behav_mask],
        'type': events['type'][behav_mask],
        'time_scope': events['time'][behav_mask],  # Same as time for miniscope
    }

    return sync_struct, nidq_struct


def extract_reward_events(
    nidq_struct: Dict[str, Any],
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract reward event times from NIDQ structure.

    Args:
        nidq_struct: NIDQ structure from process_nidq_signals
        config: Configuration dictionary

    Returns:
        Tuple of (left_reward_times, right_reward_times)
    """
    reward_left_channel = config['nidq']['reward_left_bit'] + 1
    reward_right_channel = config['nidq']['reward_right_bit'] + 1

    # Extract rising edges for rewards
    left_mask = (nidq_struct['chan'] == reward_left_channel) & (nidq_struct['type'] == 1)
    right_mask = (nidq_struct['chan'] == reward_right_channel) & (nidq_struct['type'] == 1)

    return nidq_struct['time'][left_mask], nidq_struct['time'][right_mask]
