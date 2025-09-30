# NATO hackathon - Phase 1

This repository contains a single Jupyter notebook implementing a workflow for semi-supervised labeling and CNN-based classification of **DroneRFb-Spectra** data.  

> **Note:** The dataset and trained model are **not included** due to large file sizes. The dataset can be downloaded from the [References](#references) section below.

## Overview

The notebook covers the following steps:

1. **Prepare files**  
   - Load `.npy` spectrogram files from local directories.  
   - Visualize sample spectrograms.  
   - Normalize each sample for better contrast.

2. **Labeling**  
   - Dataset is already labeled via folder names.  
   - Tested semi-supervised labeling techniques:  
     - **KMeans label propagation**  
     - **LabelPropagation** (from `sklearn`)  
   - Achieved up to ~60% accuracy using LabelPropagation on a subset.

3. **Cross-validation**  
   - 5-fold stratified cross-validation.  
   - Visualizes class distributions in train/validation splits.

4. **Dataset Class**  
   - `SpectrogramDataset` handles loading spectrograms as PyTorch tensors.  
   - Adds channel dimension for CNN input.

5. **CNN Model**  
   - `SpectrogramCNN` with 3 convolutional layers and 2 fully connected layers.  
   - Input size: `(1, 543, 512)` → output: 24 classes.  

6. **Training**  
   - Uses weighted sampling for imbalanced classes.  
   - Adam optimizer, CrossEntropy loss.  
   - 5 epochs per fold (short for demonstration).  
   - Saves final model and checkpoint.

7. **Metrics**  
   - Accuracy, F1-macro, F1-weighted are calculated for each fold.  
   - Average cross-validation performance reported.

## Requirements

```bash
numpy
matplotlib
seaborn
torch
scikit-learn
```

## Usage

  1. Adjust root_dir to point to your local dataset.

  2. Run all notebook cells sequentially to:

  3. Load and preprocess data

  4. Perform semi-supervised labeling

  5. Train CNN model

  6. Evaluate metrics

  7. Model checkpoints will be saved locally.

## References

Dataset from Ningning Yu, *"DroneRFb-Spectra: A RF spectrogram dataset for drone recognition"*, IEEE Dataport, May 23, 2024, [doi:10.21227/wv7h-sv64](https://doi.org/10.21227/wv7h-sv64)
