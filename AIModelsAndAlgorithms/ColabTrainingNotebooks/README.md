# Colab Training Notebooks

These notebooks are for training the app-facing models on Google Colab and
saving artifacts back to Google Drive.

Before running, upload the datasets to:

```text
/content/drive/MyDrive/VisualizationApp/Data
```

The notebooks now clone the project code from GitHub into Colab:

```text
/content/VisualizationApp
```

Then they symlink the Drive datasets into the cloned repo's `Data/` folder so
the existing training scripts can read `Data/*.csv`.

In each notebook, edit:

```python
GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/VisualizationApp.git"
GITHUB_BRANCH = "main"
```

Training outputs are saved back to:

```text
/content/drive/MyDrive/VisualizationApp/AIModelsAndAlgorithms/...
```

If your new datasets are in another Drive folder, edit `NEW_DATA_DIR` in the
notebook and run the optional copy cell.

## Notebooks

| Notebook | Model | Saves To |
|---|---|---|
| `01_occupancy_random_forest_colab.ipynb` | Occupancy RandomForest | `AIModelsAndAlgorithms/OccupancyPrediction` |
| `02_lighting_persona_random_forest_colab.ipynb` | Lighting persona sliding-window HistGradientBoostingClassifier | `AIModelsAndAlgorithms/LightingPersona` |
| `03_lighting_recommendation_hgb_colab.ipynb` | Lighting recommendation HistGradientBoostingRegressor | `AIModelsAndAlgorithms/LightingRecommendation` |
| `04_temperature_recommendation_hgb_energy_aware_colab.ipynb` | Temperature recommendation energy-aware HistGradientBoostingRegressor | `AIModelsAndAlgorithms/TempretureRecomendation` |
| `05_lighting_persona_transformer_colab_runner.ipynb` | Lighting persona Transformer | `AIModelsAndAlgorithms/LightingPersona/transformer` |
| `06_temperature_recommendation_transformer_energy_aware_colab.ipynb` | Temperature recommendation energy-aware Transformer | `AIModelsAndAlgorithms/TempretureRecomendation/transformer` |
| `07_occupancy_transformer_colab.ipynb` | Occupancy Transformer | `AIModelsAndAlgorithms/OccupancyPrediction/trandformer` |
| `08_lighting_recommendation_transformer_colab.ipynb` | Lighting recommendation Transformer | `AIModelsAndAlgorithms/LightingRecommendation/transformer` |
| `09_temperature_persona_transformer_colab.ipynb` | Temperature persona Transformer | `AIModelsAndAlgorithms/TempreturePersona/transformer` |
| `10_temperature_persona_hgb_colab.ipynb` | Temperature persona HistGradientBoostingClassifier | `AIModelsAndAlgorithms/TempreturePersona` |

## Recommended Flow

1. Open one notebook in Colab.
2. Use GPU runtime for Transformer notebooks.
3. Run the setup cells.
4. Run the smoke-test cell first.
5. Run the full-training cell.
6. Check the final artifact listing cell.

The classical machine learning notebooks also run in Colab. They do not need a
GPU, but Colab can still help because the new datasets are large.

## Accuracy Fixes Added

- `02_lighting_persona_random_forest_colab.ipynb` now trains the app-facing
  `lighting_persona_model.joblib` from sliding 4-hour windows instead of one
  room-day sample. This should create many more usable samples from the same
  two-year lighting dataset.
- `05_lighting_persona_transformer_colab_runner.ipynb` now trains on rolling
  2-hour sequences (`sequence-length 24`) instead of full-day room samples.
- `06_temperature_recommendation_transformer_energy_aware_colab.ipynb` now uses
  larger defaults for full training (`2,000,000` rows and up to `300,000`
  sequences) so the Transformer is not trained on only 1,000 sequences.
