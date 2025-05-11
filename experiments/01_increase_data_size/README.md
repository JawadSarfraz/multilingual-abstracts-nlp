# Experiment 01: Increase Data Size

## Objective:
- To assess the impact of increasing the dataset size to 20k samples on model performance.

## Steps:
1. Extract a sample of 20k objects from the main dataset.
2. Preprocess the data using the existing data preprocessing pipeline.
3. Train the model using the expanded dataset.
4. Evaluate the model and compare results with the baseline.

## Dataset:
- Source: `data/raw/data.json`
- Target: `data/processed/20k_sample.json`

## Metrics:
- F1-Score
- Precision
- Recall
- Hamming Loss

---

## Results:
- Initial Baseline (5k samples):
  - F1-Score: TBD
  - Precision: TBD
  - Recall: TBD
  - Hamming Loss: TBD

- Updated (20k samples):
  - F1-Score: TBD
  - Precision: TBD
  - Recall: TBD
  - Hamming Loss: TBD

## Step 1: Data Sampling
- Extracted 20k samples from the original dataset located at `data/raw/data.json`.
- The sampled data is saved at `experiments/01_increase_data_size/data_20k.json`.
- Command to run the sampling script:
  ```bash
  python3 data_sampling_20k.py

