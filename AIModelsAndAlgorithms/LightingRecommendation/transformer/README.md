# Lighting Recommendation Transformer

This folder contains the Google Colab training notebook for a Transformer-based lighting recommendation regressor.

Use `lighting_recommendation_transformer_colab.ipynb` in Colab Pro/Pro+:

1. Upload this notebook to Colab.
2. Set runtime to GPU.
3. Run all cells.
4. When Google Drive is mounted, keep `DATA_CSV` pointing to your Drive copy of `lightningData.csv`.
5. After training, download `lighting_recommendation_transformer_outputs.zip` from your Drive output folder.
6. Unzip the model files into this folder when you want the backend to use the Transformer version later.

The notebook saves:

- `lighting_recommendation_transformer.pt`
- `lighting_recommendation_transformer_metadata.json`
- `lighting_recommendation_transformer_report.txt`
- `lighting_recommendation_transformer_sample_predictions.csv`
- `lighting_recommendation_transformer_outputs.zip`

The model predicts an efficient lighting level for the next five-minute recommendation step using recent per-room/per-lamp history.
