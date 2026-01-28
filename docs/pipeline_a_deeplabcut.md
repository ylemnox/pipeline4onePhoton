# Pipeline A: Behavior Extraction with DeepLabCut

This document describes how to use DeepLabCut for behavior extraction from webcam videos. This pipeline runs **externally** and provides a CSV file as input to the main 1p-spatial-pipeline.

## Overview

DeepLabCut (DLC) is used to track an LED marker attached to the mouse, providing position data for each video frame.

**Input**: Fragmented webcam .avi videos (e.g., `./miniscope/MY01_20251231/My_WebCam/`)
**Output**: CSV file with columns (frame, x, y, likelihood)

## Provided Notebook

We provide a modified Colab notebook for users without local GPU access:

**`notebooks/SingleMouseTracking_via_DeepLabCut.ipynb`**

This notebook is adapted from the official DeepLabCut Colab demo and includes:
- Step-by-step GUI instructions for frame labeling (Step 1)
- Colab environment setup with PyTorch backend (Step 2)
- Training with ResNet-50 architecture (Step 3)
- Video analysis with outlier filtering (Step 4)
- CSV merging script for combining per-video results (Step 4)
- Optional labeled video creation (Step 5)

**Recommended workflow**: Follow Steps 1-4 below using the notebook as your primary guide, referring to this document for additional context.

## Prerequisites

- DeepLabCut installed locally (for GUI labeling)
- Google Colab account (for GPU training, if no local GPU)
- Webcam videos from your experiment

## Workflow

### Step 1: Create Project and Label Frames (Local)

> **See**: Notebook Step 1 for detailed GUI instructions with screenshots

1. Install DeepLabCut with GUI support:
   ```bash
   conda create -n dlc python=3.9 -y
   conda activate dlc
   pip install "deeplabcut[gui]"
   pip install "deeplabcut[tf]"
   ```

2. Launch the GUI:
   ```bash
   python -m deeplabcut
   ```

3. Create a new project:
   - Project name: e.g., `SingleMouseTracking`
   - Experimenter: your name
   - Select 3-5 representative videos (not all videos)
   - Check "Copy videos to project folder"

4. Configure the project:
   - Open `config.yaml` in your project folder
   - Set `bodyparts: [LED]` (remove default bodyparts)
   - Change video format to match your files (e.g., `avi`)

5. Extract and label frames:
   - Click "Extract frames" → use automatic extraction
   - Click "Label frames" → opens napari window
   - For each frame: click XOR button, click on LED position
   - Save with Ctrl+S after labeling each video's frames

### Step 2: Setup Colab and Train Network

> **See**: Notebook Steps 2-3 for complete Colab setup

If you don't have a local GPU, use Google Colab with the provided notebook:

1. Upload your **project folder** to Google Drive

2. Open `notebooks/SingleMouseTracking_via_DeepLabCut.ipynb` in Colab

3. Change runtime to GPU: Runtime → Change runtime type → GPU

4. Edit the configuration cell:
   ```python
   project_folder_name = "your_project_folder_name"
   video_type = "avi"  # match your video format
   ```

5. Create training dataset (uses PyTorch backend):
   ```python
   deeplabcut.create_training_dataset(
     path_config_file, net_type="resnet_50", engine=deeplabcut.Engine.PYTORCH
   )
   ```

6. Train the network (~200 epochs recommended):
   ```python
   deeplabcut.train_network(
       path_config_file,
       shuffle=1,
       save_epochs=5,  # saves checkpoints every 5 epochs
       epochs=200,
       batch_size=8,
   )
   ```

7. Evaluate the network:
   ```python
   deeplabcut.evaluate_network(path_config_file, plotting=True)
   ```

**Note**: If Colab disconnects, you can resume from a checkpoint using `snapshot_path` argument.

### Step 3: Analyze Videos

> **See**: Notebook Step 4 for video analysis code

```python
import glob
import os

video_folder_path = '/content/drive/MyDrive/My_WebCam'
all_videos = sorted(glob.glob(os.path.join(video_folder_path, '*.avi')))

# Analyze all videos
deeplabcut.analyze_videos(path_config_file, all_videos, save_as_csv=True)

# Apply outlier filtering (recommended)
deeplabcut.filterpredictions(path_config_file, all_videos, save_as_csv=True)
```

This creates CSV files for each video with predictions. The filtered versions (`*_filtered.csv`) have outliers corrected.

### Step 4: Merge CSV Results

> **See**: Notebook Step 4 (final cell) for the complete merging script

The output CSV files need to be merged into a single file. The notebook provides a ready-to-use script that:
- Uses filtered CSV files (`*_filtered.csv`) for better quality
- Sorts files numerically by leading number
- Preserves DLC multi-header format
- Resets frame indices for continuous indexing

```python
import pandas as pd
import glob
import os
import re

# Edit path of folder where csv files are saved
csv_folder_path = '/content/drive/MyDrive/video_folder/'

# Find filtered csv files
file_pattern = os.path.join(csv_folder_path, '*_filtered.csv')
all_files = glob.glob(file_pattern)

# Sort by leading number in filename
def get_leading_number(filepath):
    filename = os.path.basename(filepath)
    match = re.match(r'^(\d+)', filename)
    return int(match.group(1)) if match else -1

all_files.sort(key=get_leading_number)

print(f"Total {len(all_files)} files found.")

# Concatenate (preserves DLC header format)
df_list = []
for file in all_files:
    df = pd.read_csv(file, header=[0, 1, 2], index_col=0)
    df_list.append(df)

merged_df = pd.concat(df_list)
merged_df = merged_df.reset_index(drop=True)

# Save merged file
output_path = os.path.join(csv_folder_path, 'Final_Merged_Result.csv')
merged_df.to_csv(output_path)
print(f"Saved to {output_path}")
```

### Step 5: Integrate with Calcium Pipeline

1. Place the merged CSV file in your data directory

2. Update your `config.yaml`:
   ```yaml
   behavior:
     position_csv: "./data/deepLabCut_result_MY01_20251231.csv"
     likelihood_threshold: 0.9
     track_length_cm: 48
   ```

3. Run the 1p-spatial-pipeline:
   ```bash
   1p-spatial-pipeline merge --config config.yaml
   ```

## Output Format

The final CSV file should have this format:

```
scorer,DLC_Resnet50
bodyparts,LED
coords,x,y,likelihood
0,523.4,312.1,0.998
1,524.1,311.8,0.997
2,525.2,311.5,0.996
...
```

Where:
- Column 0: Frame index (continuous across all videos)
- Column 1: X position in pixels
- Column 2: Y position in pixels
- Column 3: Likelihood (0-1, confidence of prediction)

## Optional: Create Labeled Videos

> **See**: Notebook Step 5

For visualization and quality checking, you can create videos with predicted labels overlaid:

```python
deeplabcut.create_labeled_video(path_config_file, all_videos, filtered=True)
```

Check the likelihood plot (`plot-likelihood.png`) and consider adjusting `p-cutoff` in `config.yaml` (e.g., 0.8-0.9) for higher confidence visualization.

## Tips

1. **Label quality matters**: Spend time labeling frames accurately. This directly affects tracking quality.

2. **Check likelihood**: Low likelihood values indicate uncertain predictions. The pipeline interpolates these points.

3. **Consistent lighting**: Ensure your webcam lighting is consistent across recording sessions.

4. **LED visibility**: Make sure the LED marker is clearly visible and doesn't get occluded.

5. **Frame rate**: Note your webcam frame rate for proper temporal alignment in the pipeline.

6. **Use filtered predictions**: Always use `*_filtered.csv` files instead of raw predictions for better accuracy.

7. **Colab checkpoints**: Training saves checkpoints every 5 epochs. If disconnected, resume with `snapshot_path` argument.

## Troubleshooting

### Poor tracking accuracy
- Add more labeled frames, especially for difficult poses
- Increase training iterations
- Check if the LED is clearly visible in all videos

### Memory errors during training
- Reduce batch size in the DLC config
- Use Google Colab for GPU training

### CSV merging issues
- Ensure all CSV files have the same format
- Check that frame indices are continuous after merging
- Use `*_filtered.csv` files, not raw prediction files

### Colab disconnection during training
- Training saves checkpoints every 5 epochs
- Resume from checkpoint: `deeplabcut.train_network(..., snapshot_path="/path/to/snapshot-050.pt")`
- Check your Drive for `dlc-models-pytorch` folder with saved snapshots

### Model performance shows NaN during training
- This is normal at the start of training; the model is still learning
- Metrics should improve as training progresses

## References

- **Provided Notebook**: `notebooks/SingleMouseTracking_via_DeepLabCut.ipynb`
- [DeepLabCut Documentation](https://deeplabcut.github.io/DeepLabCut/)
- [DeepLabCut GitHub](https://github.com/DeepLabCut/DeepLabCut)
- [DLC Colab Notebooks](https://github.com/DeepLabCut/DeepLabCut/tree/master/examples)
- [DLC Single Animal User Guide](https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html)
