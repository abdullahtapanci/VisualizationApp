# Occupancy Transformer

This folder contains a Google Colab notebook for training a PyTorch Transformer
model for next-hour occupancy prediction.

The notebook expects these files in Google Drive:

- `PIRSensorData.csv`
- `hotelReservationData.csv`

Open this file in Colab:

```text
occupancy_transformer_colab.ipynb
```

In Colab:

1. Set `Runtime` -> `Change runtime type` -> `GPU`.
2. Run the Drive mount/data path cell.
3. Adjust `MAX_SEQUENCES`, `STRIDE`, `EPOCHS`, or `BATCH_SIZE` if needed.
4. Run training.
5. The notebook saves `occupancy_transformer_outputs.zip` to Google Drive.

Outputs inside the zip:

- `occupancy_transformer.pt`
- `occupancy_transformer_metadata.json`
- `occupancy_transformer_report.txt`
- `occupancy_transformer_confusion_matrix.csv`

