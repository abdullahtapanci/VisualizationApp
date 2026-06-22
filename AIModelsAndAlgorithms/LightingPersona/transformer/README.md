# Lighting Persona Transformer

This folder contains a PyTorch Transformer model for predicting `lightning_persona`
from room-day lighting sequences.

Each training example is one room on one day:

- 288 time steps per day, one every 5 minutes
- per-lamp lighting levels
- PIR motion
- occupant/activity features
- cyclic hour-of-day features

The model predicts the dominant `lightning_persona` for that room-day.

## Quick Smoke Test

Use a small row limit to confirm the script runs:

```bash
cd /Users/abdullahtapanci/Desktop/allFolders/Work/VisualizationApp
./venv/bin/python AIModelsAndAlgorithms/LightingPersona/transformer/lighting_persona_transformer.py --max-rows 50000 --epochs 2
```

## Full Training

```bash
cd /Users/abdullahtapanci/Desktop/allFolders/Work/VisualizationApp
./venv/bin/python AIModelsAndAlgorithms/LightingPersona/transformer/lighting_persona_transformer.py
```

## Google Colab GPU Training

Open this notebook in Colab:

```text
lighting_persona_transformer_colab.ipynb
```

In Colab:

1. Set `Runtime` -> `Change runtime type` -> `GPU`.
2. Upload `Data/lightningData.csv`, or mount Google Drive and set `DATA_CSV`.
3. Run the notebook cells.
4. Download `lighting_persona_transformer_outputs.zip`.
5. Unzip the files back into this local folder.

Outputs:

- `lighting_persona_transformer.pt`
- `lighting_persona_transformer_metadata.json`
- `lighting_persona_transformer_report.txt`
- `lighting_persona_transformer_confusion_matrix.csv`
