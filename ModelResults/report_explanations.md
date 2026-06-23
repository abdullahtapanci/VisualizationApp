# AI Model Result Plot Explanations

This folder contains result plots for the main AI models used in the visualization app. The plots are grouped by model under `ModelResults/`. Classification models include confusion matrices, per-class precision/recall/F1 plots, and class support plots. Regression and recommendation models include metric summaries, prediction-vs-target plots, residual distributions, group-level error plots, and energy-saving summaries where applicable.

## Occupancy Random Forest

The Occupancy Random Forest reaches high overall accuracy (`0.974`) and a strong weighted F1 score (`0.981`). This means the model performs well for the dominant room states, especially `Occupied` and `Vacant`, which have large test support and F1 scores of `0.990` and `0.977`. The confusion matrix plots show that most examples are correctly separated between occupied and vacant states. However, the `Cleaning` class is weak: its F1 score is only `0.150`, mostly because the class has very low support compared with the other states. Therefore, this model is reliable for general occupancy detection, but not strong enough for accurately identifying cleaning events.

Recommended plots for the report:
- `ModelResults/occupancy_random_forest/02_confusion_matrix_counts.png`
- `ModelResults/occupancy_random_forest/03_confusion_matrix_percent.png`
- `ModelResults/occupancy_random_forest/04_per_class_precision_recall_f1.png`

## Occupancy Transformer

The Occupancy Transformer performs slightly better than the Random Forest overall, with accuracy `0.987` and weighted F1 `0.989`. The model was trained on `400,000` temporal sequences, so it can use recent room history rather than only static row-level features. The plots show very strong classification for `Occupied` and `Vacant`, both near `0.99` F1. Like the Random Forest, the main weakness is still `Cleaning`, with F1 `0.167`. This suggests that the temporal model improves the main occupancy states but still needs more cleaning examples or better cleaning-specific features.

Recommended plots for the report:
- `ModelResults/occupancy_transformer/01_metric_summary.png`
- `ModelResults/occupancy_transformer/03_confusion_matrix_percent.png`
- `ModelResults/occupancy_transformer/04_per_class_precision_recall_f1.png`

## Lighting Persona HistGradientBoostingClassifier

The lighting persona HGB model is one of the strongest classification models in the project. It reaches accuracy `0.9877`, macro F1 `0.9885`, and weighted F1 `0.9878`. All lighting persona classes are predicted well, including `NightFocused`, which has F1 `0.9951`. This indicates that the sliding-window feature design captures lighting behavior clearly, including lamp usage, brightness levels, motion, occupants, and time-of-day patterns. The current model artifact is identified in the metadata as a `hist_gradient_boosting_classifier`.

Recommended plots for the report:
- `ModelResults/lighting_persona_hgb/03_confusion_matrix_percent.png`
- `ModelResults/lighting_persona_hgb/04_per_class_precision_recall_f1.png`
- `ModelResults/lighting_persona_hgb/05_class_support.png`

## Lighting Persona Transformer

The Lighting Persona Transformer is weak compared with the HGB model. Its accuracy is `0.500`, macro F1 is about `0.415`, and weighted F1 is `0.500`. The main issue is the `NightFocused` class, which has F1 only `0.020`. The confusion matrix shows that many `NightFocused` examples are confused with `StaticDim`, meaning the Transformer struggles to distinguish night-specific lighting behavior from generally dim lighting. This model should not be treated as the preferred lighting persona classifier in the report; the HGB model is much more reliable for this task.

Recommended plots for the report:
- `ModelResults/lighting_persona_transformer/01_metric_summary.png`
- `ModelResults/lighting_persona_transformer/03_confusion_matrix_percent.png`
- `ModelResults/lighting_persona_transformer/04_per_class_precision_recall_f1.png`

## Lighting Recommendation HistGradientBoostingRegressor

The lighting recommendation HGB model performs well as a brightness recommendation model. It achieves MAE `1.055`, RMSE `3.284`, and R2 `0.962`, showing that predicted brightness levels are close to the efficient target values. The model also produces an estimated `39.57%` saving compared with actual lighting usage, slightly higher than the efficient target saving of `35.98%`. This suggests that the model learns an energy-saving brightness policy while still following the generated target behavior.

Recommended plots for the report:
- `ModelResults/lighting_recommendation_hgb/01_metric_summary.png`
- `ModelResults/lighting_recommendation_hgb/02_prediction_vs_target.png`
- `ModelResults/lighting_recommendation_hgb/04_actual_target_model_mean.png`

## Lighting Recommendation Transformer

The Lighting Recommendation Transformer is the strongest lighting recommendation model. It reaches MAE `0.649`, RMSE `1.570`, and R2 `0.993`, which means the predicted brightness levels match the target recommendations very closely. Its model saving is `32.73%` compared with actual usage, almost identical to the target saving of `32.32%`. This is a strong result because the Transformer uses recent lighting history and context, allowing it to model temporal behavior more accurately than the tabular HGB version.

Recommended plots for the report:
- `ModelResults/lighting_recommendation_transformer/01_metric_summary.png`
- `ModelResults/lighting_recommendation_transformer/02_prediction_vs_target.png`
- `ModelResults/lighting_recommendation_transformer/03_residual_distribution.png`
- `ModelResults/lighting_recommendation_transformer/04_actual_target_model_mean.png`

## Temperature Persona HistGradientBoostingClassifier

The temperature persona HGB model performs well overall, with accuracy `0.9000`, macro F1 `0.8951`, and weighted F1 `0.8962`. It predicts `EnergySaver`, `Housekeeping`, and `Preconditioning` almost perfectly in the test set. The main weakness is `AlwaysOnComfort`, with F1 `0.5777`, meaning this class is sometimes confused with similar comfort behavior such as `Reactive`. Overall, this model is reliable enough for use in the app, but the report should mention that comfort-focused personas are harder to separate than the more rule-like energy-saving or housekeeping states.

Recommended plots for the report:
- `ModelResults/temperature_persona_hgb/03_confusion_matrix_percent.png`
- `ModelResults/temperature_persona_hgb/04_per_class_precision_recall_f1.png`
- `ModelResults/temperature_persona_hgb/07_confidence_distribution.png`

## Temperature Persona Transformer

The Temperature Persona Transformer is acceptable but weaker than the HGB model. It reaches accuracy `0.8103`, macro F1 `0.8826`, and weighted F1 `0.8141`. The model performs very well for `Housekeeping` and `Preconditioning`, both near perfect F1, and it also performs reasonably for `AlwaysOnComfort` and `EnergySaver`. The weakest class is `Reactive`, with F1 `0.7737`, caused by many examples being predicted as reactive behavior. This model is usable, but the HGB model provides a stronger and more stable result for temperature persona prediction.

Recommended plots for the report:
- `ModelResults/temperature_persona_transformer/01_metric_summary.png`
- `ModelResults/temperature_persona_transformer/03_confusion_matrix_percent.png`
- `ModelResults/temperature_persona_transformer/04_per_class_precision_recall_f1.png`

## Temperature Recommendation HistGradientBoostingRegressor

The temperature recommendation HGB model is very strong. It achieves MAE `0.039°C`, RMSE `0.346°C`, and R2 `0.989`, meaning recommended setpoints are extremely close to the energy-aware comfort-constrained target values. The model produces an estimated `5.90%` energy saving compared with the current setpoint behavior, slightly above the target saving of `5.53%`. The mean comfort gap is almost the same as the target (`5.289°C` vs `5.278°C`), indicating that energy reduction is achieved without meaningfully increasing comfort deviation relative to the target strategy.

Recommended plots for the report:
- `ModelResults/temperature_recommendation_hgb/01_metric_summary.png`
- `ModelResults/temperature_recommendation_hgb/02_prediction_vs_target.png`
- `ModelResults/temperature_recommendation_hgb/06_energy_proxy_totals.png`
- `ModelResults/temperature_recommendation_hgb/07_energy_saving_percent.png`

## Temperature Recommendation Transformer

The temperature recommendation Transformer is also strong after energy-aware training. It reaches MAE `0.126°C`, RMSE `0.576°C`, and R2 `0.974`. The estimated model saving is `8.77%`, close to the target saving of `9.21%`, while the model comfort gap (`4.185°C`) remains close to the target comfort gap (`4.221°C`). This shows that the Transformer learned the main energy-saving setpoint strategy while preserving the comfort balance defined in the training target. It is slightly less accurate than the HGB model, but it has the advantage of using recent sequential temperature history.

Recommended plots for the report:
- `ModelResults/temperature_recommendation_transformer/01_metric_summary.png`
- `ModelResults/temperature_recommendation_transformer/02_prediction_vs_target.png`
- `ModelResults/temperature_recommendation_transformer/06_energy_proxy_totals.png`
- `ModelResults/temperature_recommendation_transformer/07_energy_saving_percent.png`

## Overall Interpretation

The strongest models are the Lighting Recommendation Transformer, Temperature Recommendation HGB, Temperature Recommendation Transformer, Lighting Persona HGB, and Occupancy Transformer. The main weak model is the Lighting Persona Transformer, which does not reliably learn the `NightFocused` persona. The occupancy models are strong for normal `Occupied` and `Vacant` states, but weak for the rare `Cleaning` class. For deployment and report conclusions, the HGB lighting persona model should be preferred over the lighting persona Transformer, while both recommendation models can be presented as successful energy-aware prediction models.
