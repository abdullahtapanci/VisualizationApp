# Tempreture Recomendation Transformer

This folder contains a Google Colab notebook and a local Python trainer for a
Transformer regressor that recommends an HVAC setpoint from room temperature
history, occupancy state, and temperature persona.

The current local trainer uses an energy-aware comfort-constrained target. It
scores candidate setpoints using estimated HVAC energy, comfort deviation, and
setpoint-change cost, then trains the Transformer to predict the best setpoint.

The notebook uses `temperatureData.csv` from Google Drive. During training, it uses existing dataset columns as stand-ins for upstream model outputs:

- `room_state` -> occupancy prediction
- `ac_persona` -> temperature persona prediction

Open in Colab:

```text
tempreture_recomendation_transformer_colab.ipynb
```

Local smoke test:

```bash
./venv/bin/python AIModelsAndAlgorithms/TempretureRecomendation/transformer/train_energy_aware_transformer.py \
  --max-rows 10000 \
  --max-sequences 5000 \
  --sequence-length 6 \
  --epochs 2 \
  --output-dir /private/tmp/temperature_transformer_smoke
```

Local full training:

```bash
./venv/bin/python AIModelsAndAlgorithms/TempretureRecomendation/transformer/train_energy_aware_transformer.py \
  --max-sequences 500000 \
  --epochs 20
```

Recommended Colab steps:

1. Set `Runtime` -> `Change runtime type` -> `GPU`.
2. Run the Drive mount cell.
3. Confirm `DATA_CSV` points to your Drive copy of `temperatureData.csv`.
4. Run all cells.
5. Download or copy `tempreture_recomendation_transformer_outputs.zip` from Drive.

Outputs:

- `tempreture_recomendation_transformer.pt`
- `tempreture_recomendation_transformer_metadata.json`
- `tempreture_recomendation_transformer_report.txt`
- `tempreture_recomendation_transformer_sample_predictions.csv`
- `tempreture_recomendation_transformer_outputs.zip`
