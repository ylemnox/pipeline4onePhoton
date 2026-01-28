# Pipeline B: CNMF-E Processing with CaImAn

This document describes how to run Pipeline B for calcium source extraction from miniscope videos using the provided Jupyter notebook.

## Overview

Pipeline B uses CaImAn's CNMF-E (Constrained Non-negative Matrix Factorization for Endoscopic data) algorithm to extract calcium signals from one-photon miniscope recordings, followed by OASIS for spike deconvolution.

**Input**: Fragmented miniscope .avi videos (e.g., `./miniscope/MY01_20251231/My_V4_Miniscope/`)
**Output**: MAT file with calcium traces (C), deconvolved spikes (S), and spatial footprints
## Recommended Workflow: Local Conda Environment

To run locally or need to process many datasets, you can install CaImAn via conda.

### Prerequisites

- Anaconda or Miniconda installed
- Sufficient RAM (16GB+ recommended)
- Multi-core CPU (processing is parallelized)

### Installation
- For more detail, visit CaImAn Repository using link in [References](#references)

```bash
# Create a new conda environment with Python 3.10
conda create -n caiman python=3.10 -y
conda activate caiman

# Install CaImAn from conda-forge
conda install -c conda-forge caiman -y

# Verify installation
python -c "import caiman; print(f'CaImAn version: {caiman.__version__}')"
```

### Running the Notebook Locally

```bash
conda activate caiman
jupyter notebook notebooks/cnmfE_pipeline.ipynb
```
## Alternative Workflow: Google Colab
If you prefer to run Pipeline B is via Google Colab, which provides free GPU access and avoids local CaImAn installation.

### Step 1: Open the Notebook

Open `notebooks/cnmfE_pipeline.ipynb` in Google Colab:
1. Go to [Google Colab](https://colab.research.google.com/)
2. File -> Upload notebook -> Select `cnmfE_pipeline.ipynb`
3. Or upload the notebook to Google Drive and open from there

### Step 2: Upload Your Data

Upload your miniscope .avi video files to Colab or mount your Google Drive:

```python
# Mount Google Drive (recommended for large files)
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Run the Notebook

Follow the notebook cells to:
1. Install CaImAn (automated in notebook)
2. Configure parameters for your data
3. Run CNMF-E source extraction
4. Run OASIS spike deconvolution
5. Export results to MAT file

### Step 4: Download the Output

Download the output MAT file to use with the downstream pipeline:

```python
from google.colab import files
files.download('MY01_20251231_data.mat')
```

### Step 5: Run Downstream Pipelines

Add the MAT file path to your `config.yaml`:

```yaml
cnmfe:
  mat_file: "/path/to/cnmfe_output.mat"
```

Then run the downstream pipelines:

```bash
# Merge data (Pipeline D)
pipeline4onePhoton merge --config config.yaml

# Or run all downstream pipelines
pipeline4onePhoton run-all --config config.yaml --mode both
```

Note: You can also use `--cnmfe-mat` CLI flag to override the config value.


## Parameter Tuning

Key CNMF-E parameters to adjust based on your data:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gSig` | [4, 4] | Gaussian kernel half-width (pixels) - adjust based on cell size |
| `gSiz` | [13, 13] | Average neuron diameter (pixels) |
| `min_corr` | 0.8 | Minimum local correlation for cell detection (0-1) |
| `min_pnr` | 10 | Minimum peak-to-noise ratio |
| `ssub` | 2 | Spatial downsampling factor |
| `tsub` | 2 | Temporal downsampling factor |
| `p` | 1 | AR model order (1 for fast dynamics, 2 for slower) |

### Tips

1. **gSig and gSiz**: Measure typical cell diameter in pixels from your videos. `gSig` should be roughly half the cell radius.

2. **min_corr and min_pnr**: Start with defaults, then adjust:
   - Too many false positives -> increase these values
   - Missing real cells -> decrease these values

3. **ssub/tsub**: Increase for faster processing and lower memory, but may lose small/fast signals.

4. **p**: Use `p=1` for GCaMP6f (fast), `p=2` for GCaMP6s (slow).

## Output Format

The notebook creates a MAT file with this structure:

```
{experiment_name}_data.mat
└── calcium (struct)
    ├── C                 # Denoised calcium traces (n_frames x n_neurons)
    ├── S                 # Deconvolved spikes (n_frames x n_neurons)
    ├── F_dff             # dF/F traces (if computed)
    ├── footprint         # Spatial footprints (n_neurons x height x width)
    ├── footprint_center  # Neuron centroids (n_neurons x 2) [y, x]
    ├── idx_components    # Indices of accepted components
    ├── idx_components_bad # Indices of rejected components
    ├── SNR_comp          # SNR for each component
    ├── r_values          # Spatial correlation values
    ├── cnn_preds         # CNN classifier predictions
    ├── neurons_sn        # Noise level for each neuron
    ├── Cn                # Correlation image
    ├── pnr               # Peak-to-noise ratio image
    ├── n_frames          # Total number of frames
    ├── n_neuron          # Total number of neurons
    ├── frame_rate        # Recording frame rate
    └── fov_size          # Field of view dimensions [height, width]
```

**Required fields** for downstream pipelines:
- `calcium.C` - Calcium traces (n_frames x n_neurons)
- `calcium.S` - Deconvolved spikes (n_frames x n_neurons)
- `calcium.footprint_center` - Neuron centers (n_neurons x 2)
- `calcium.idx_components` - Good component indices

## Troubleshooting

### Memory Error During Processing

- Increase `ssub` and `tsub` values (e.g., from 2 to 4)
- Reduce `rf` (patch size)
- Process fewer videos at once
- Use Colab Pro for more RAM

### No Components Found

- Decrease `min_corr` and `min_pnr` thresholds
- Check that `gSig` matches your cell size
- Verify videos have good signal (check raw videos)

### Too Many False Positives

- Increase `min_corr` (try 0.85-0.9)
- Increase `min_pnr` (try 12-15)
- Increase `min_SNR` (try 4-5)
- Enable CNN filtering: `use_cnn: true`

## Processing Time

Typical processing times (varies by hardware and data size):
- 10,000 frames, 512x512 pixels: 30-60 minutes
- 50,000 frames, 512x512 pixels: 2-4 hours

Google Colab with GPU can significantly speed up processing.

## References

- [CaImAn GitHub](https://github.com/flatironinstitute/CaImAn)
- [CaImAn Documentation](https://caiman.readthedocs.io/)
- [CNMF-E Paper](https://elifesciences.org/articles/28728) - Zhou et al., 2018
- [OASIS Paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423) - Friedrich et al., 2017

## License

The CNMF-E notebook uses CaImAn which is licensed under GPL-2.0. See `notebooks/LICENSE.GPL` for details.
