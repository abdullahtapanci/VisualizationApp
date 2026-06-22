"""Per-stay feature extraction for guest segmentation.

A "stay" = one guest_id in PIRSensorData.csv. We summarize the room's
5-minute motion time series into a fixed-length feature vector and
join in a few reservation attributes.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent.parent / "Data"


def build_stay_features():
    pir = pd.read_csv(DATA / "PIRSensorData.csv", parse_dates=["timestamp"])
    pir = pir.dropna(subset=["guest_id"]).copy()
    pir["guest_id"] = pir["guest_id"].astype(int)
    pir["hour"] = pir["timestamp"].dt.hour

    g = pir.groupby("guest_id")
    feats = pd.DataFrame(index=g.size().index.rename("guest_id"))
    feats["n_samples"] = g.size()
    feats["motion_rate"] = g["pir_motion"].mean()
    feats["motion_total"] = g["pir_motion"].sum()
    feats["occupied_rate"] = g["room_state"].apply(lambda s: (s == "Occupied").mean())

    # mean motion within each part of the day
    def part(mask, name):
        feats[name] = (
            pir[mask].groupby("guest_id")["pir_motion"].mean()
            .reindex(feats.index).fillna(0)
        )

    part((pir["hour"] >= 22) | (pir["hour"] < 6), "motion_night")
    part((pir["hour"] >= 6) & (pir["hour"] < 12), "motion_morning")
    part((pir["hour"] >= 12) & (pir["hour"] < 18), "motion_afternoon")
    part((pir["hour"] >= 18) & (pir["hour"] < 22), "motion_evening")

    # 24-bucket hour-of-day motion distribution, row-normalized
    hourly = (
        pir.groupby(["guest_id", "hour"])["pir_motion"].sum()
        .unstack(fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )
    hourly = hourly.div(hourly.sum(axis=1).replace(0, 1), axis=0)
    hourly.columns = [f"h{h:02d}" for h in hourly.columns]
    feats = feats.join(hourly, how="left").fillna(0)

    # ground-truth synthetic persona (one per stay)
    feats["true_persona"] = g["persona"].agg(lambda s: s.mode().iat[0])

    # reservation features
    res = pd.read_csv(
        DATA / "hotelReservationData.csv",
        usecols=["Guest ID", "Total Nights", "Total Amount", "Adults",
                 "Children", "Room Type", "Stay_Duration"],
    ).rename(columns={"Guest ID": "guest_id"}).set_index("guest_id")
    feats = feats.join(res, how="left")
    feats["price_per_night"] = feats["Total Amount"] / feats["Stay_Duration"].replace(0, np.nan)

    return feats


def numeric_matrix(feats):
    """Return (X, feature_names) — numeric feature matrix without labels."""
    drop = {"true_persona", "Room Type"}
    cols = [c for c in feats.columns if c not in drop]
    X = feats[cols].select_dtypes(include=[np.number]).fillna(0)
    return X.values, list(X.columns)
