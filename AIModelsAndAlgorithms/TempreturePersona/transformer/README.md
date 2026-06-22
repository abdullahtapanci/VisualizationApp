# Tempreture Persona Transformer

This folder contains a Google Colab notebook for training a Transformer classifier that predicts temperature/HVAC persona labels from `temperatureData.csv`.

The notebook uses the `ac_persona` column as the default label. It mounts Google Drive, trains on room-level time-series sequences, and saves the trained model outputs back to Drive.

Open in Colab:

```text
tempreture_persona_transformer_colab.ipynb
```

Recommended Colab steps:

1. Set `Runtime` -> `Change runtime type` -> `GPU`.
2. Run the Drive mount cell.
3. Confirm `DATA_CSV` points to your Drive copy of `temperatureData.csv`.
4. Run all cells.
5. Download or copy `tempreture_persona_transformer_outputs.zip` from Drive.

Outputs:

- `tempreture_persona_transformer.pt`
- `tempreture_persona_transformer_metadata.json`
- `tempreture_persona_transformer_report.txt`
- `tempreture_persona_transformer_confusion_matrix.csv`
- `tempreture_persona_transformer_outputs.zip`
