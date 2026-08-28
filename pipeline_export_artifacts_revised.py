"""
Pipeline sinkronisasi artifact deployment Hybrid SARIMAX + XGBoost Residual.

Fokus revisi:
1. Membaca langsung struktur model dari hybrid_twp90_model.joblib dan preprocessing_config.json.
2. Tidak melakukan retraining otomatis, sehingga model final dari joblib tidak berubah.
3. Menyelaraskan metadata deployment dengan logika one-step-ahead:
   input variabel eksternal periode t -> prediksi TWP90 periode t+forecast_horizon.
4. Menyimpan metadata UI prediksi berbasis kalender agar app.py tidak memakai list/dropdown periode.
5. Mengekspor ulang config, joblib, dan file evaluasi pendukung dashboard.
6. Jika raw_history.csv tersedia, file tersebut divalidasi dan ikut diekspor. Jika belum tersedia,
   pipeline tetap berjalan, tetapi prediksi dashboard membutuhkan raw_history.csv lengkap.

Cara pakai:
    python pipeline_export_artifacts.py

Opsional:
    MODEL_ARTIFACT_DIR=./model_artifacts MODEL_ARTIFACT_OUT_DIR=./model_artifacts python pipeline_export_artifacts.py
"""

from __future__ import annotations

import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

DEFAULT_TARGET = "TWP90 (%)"
DEFAULT_DATE_COL = "Month"
DEFAULT_FORECAST_HORIZON = 1
RESIDUAL_TARGET_DEFAULT = "Residual_SARIMAX_Log"
PERCENT_COL_CANDIDATES = [
    "TWP90 (%)",
    "BI-7Day-RR",
    "Inflasi",
    "Pertumbuhan Outstanding (YoY% atau MoM%)",
]

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DEFAULT_ARTIFACT_DIR = BASE_DIR / "model_artifacts"
SOURCE_ARTIFACT_DIR = Path(os.environ.get("MODEL_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR))
OUT_DIR = Path(os.environ.get("MODEL_ARTIFACT_OUT_DIR", SOURCE_ARTIFACT_DIR))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_input_file(filename: str) -> Path:
    """Cari file di MODEL_ARTIFACT_DIR, model_artifacts, lalu folder script."""
    candidates = [SOURCE_ARTIFACT_DIR / filename, DEFAULT_ARTIFACT_DIR / filename, BASE_DIR / filename]
    seen: set[Path] = set()
    unique_candidates = []
    for path in candidates:
        if path not in seen:
            unique_candidates.append(path)
            seen.add(path)
    for path in unique_candidates:
        if path.exists():
            return path
    return unique_candidates[0]


MODEL_PATH = resolve_input_file("hybrid_twp90_model.joblib")
CONFIG_PATH = resolve_input_file("preprocessing_config.json")
RAW_HISTORY_PATH = resolve_input_file("raw_history.csv")


def json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date())
    return str(obj)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def month_end(value) -> pd.Timestamp:
    return pd.to_datetime(value).to_period("M").to_timestamp("M")


def next_month(value: pd.Timestamp, step: int = 1) -> pd.Timestamp:
    return (month_end(value).to_period("M") + int(step)).to_timestamp("M")


def parse_raw_history(path: Path, date_col: str, target: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Kolom tanggal '{date_col}' tidak ditemukan pada {path.name}.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    df.index.name = "Month"
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if target not in df.columns:
        raise ValueError(f"Kolom target '{target}' tidak ditemukan pada {path.name}.")
    return df


def dataframe_from_artifact(artifacts: dict[str, Any], key: str) -> pd.DataFrame:
    df = artifacts.get(key)
    if isinstance(df, pd.DataFrame) and not df.empty:
        out = df.copy()
        if "Month" not in out.columns:
            out = out.reset_index()
        out["Month"] = pd.to_datetime(out["Month"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
        return out.dropna(subset=["Month"]).sort_values("Month")
    return pd.DataFrame()


def export_test_predictions(artifacts: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    test_df = dataframe_from_artifact(artifacts, "test_results_hybrid")
    if not test_df.empty:
        test_df.to_csv(out_dir / "test_predictions_hybrid.csv", index=False)
    return test_df


def export_dashboard_history(raw_history: pd.DataFrame, test_predictions: pd.DataFrame, target: str, split_date: str, out_dir: Path) -> pd.DataFrame:
    if not raw_history.empty:
        dash = raw_history[[target]].copy().rename(columns={target: "Actual_Original"}).reset_index()
        dash["Data_Type"] = np.where(pd.to_datetime(dash["Month"]) < pd.to_datetime(split_date), "Train", "Test")
        if not test_predictions.empty and "Prediksi_Hybrid_Original" in test_predictions.columns:
            pred = test_predictions[["Month", "Prediksi_Hybrid_Original"]].copy()
            dash = dash.merge(pred, on="Month", how="left")
    elif not test_predictions.empty and "Actual_Original" in test_predictions.columns:
        keep_cols = [c for c in ["Month", "Actual_Original", "Prediksi_Hybrid_Original"] if c in test_predictions.columns]
        dash = test_predictions[keep_cols].copy()
        dash["Data_Type"] = "Test"
    else:
        dash = pd.DataFrame()

    if not dash.empty:
        dash = dash.sort_values("Month")
        dash.to_csv(out_dir / "dashboard_twp90_history.csv", index=False)
    return dash


def infer_last_observed(raw_history: pd.DataFrame, artifacts: dict[str, Any], cfg: dict[str, Any]) -> pd.Timestamp | None:
    if cfg.get("last_observed_month"):
        return month_end(cfg["last_observed_month"])
    if not raw_history.empty:
        return month_end(raw_history.index.max())
    residual = artifacts.get("final_residual_train_log")
    if isinstance(residual, pd.Series) and len(residual.index) > 0:
        return month_end(residual.index.max())
    test_df = dataframe_from_artifact(artifacts, "test_results_hybrid")
    if not test_df.empty:
        return month_end(test_df["Month"].max())
    return None


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"hybrid_twp90_model.joblib tidak ditemukan. Dicari di: {MODEL_PATH}")

    cfg = load_json(CONFIG_PATH)
    artifacts: dict[str, Any] = joblib.load(MODEL_PATH)

    target = cfg.get("target") or artifacts.get("target") or DEFAULT_TARGET
    date_col = cfg.get("date_col", DEFAULT_DATE_COL)
    split_date = cfg.get("split_date", "2025-01-01")
    forecast_horizon = int(cfg.get("forecast_horizon", artifacts.get("forecast_horizon", DEFAULT_FORECAST_HORIZON)))
    forecast_horizon = max(1, forecast_horizon)

    differencing_orders = artifacts.get("differencing_orders_exog") or cfg.get("differencing_orders_exog") or {}
    exog_lags_config = artifacts.get("exog_lags_config") or cfg.get("exog_lags_config") or {}
    raw_input_columns = [c for c in differencing_orders.keys() if c != target]
    percent_columns = [c for c in PERCENT_COL_CANDIDATES if c == target or c in raw_input_columns]

    raw_history = parse_raw_history(RAW_HISTORY_PATH, date_col, target)
    if not raw_history.empty:
        missing_raw_cols = [col for col in raw_input_columns if col not in raw_history.columns]
        if missing_raw_cols:
            raise ValueError("raw_history.csv belum memuat kolom eksternal: " + ", ".join(missing_raw_cols))
        raw_history.reset_index().to_csv(OUT_DIR / "raw_history.csv", index=False)

    test_predictions = export_test_predictions(artifacts, OUT_DIR)
    dashboard_history = export_dashboard_history(raw_history, test_predictions, target, split_date, OUT_DIR)

    last_observed = infer_last_observed(raw_history, artifacts, cfg)
    next_input = next_month(last_observed, 1) if last_observed is not None else None
    next_prediction = next_month(next_input, forecast_horizon) if next_input is not None else None

    enriched_config = {
        **cfg,
        "artifact_version": "joblib_json_aligned_one_step_v2",
        "deployment_mode": "one_step_latest_exog_input",
        "description": (
            "Deployment mengikuti hybrid_twp90_model.joblib dan preprocessing_config.json. "
            "Dashboard bisa memilih periode input future. Untuk periode setelah bulan pertama, input eksternal harus diisi berurutan agar fitur lag/differencing tetap konsisten."
        ),
        "date_col": date_col,
        "target": target,
        "forecast_horizon": forecast_horizon,
        "forecasting_scheme": cfg.get(
            "forecasting_scheme",
            artifacts.get("forecasting_scheme", "Hybrid SARIMAX-XGBoost residual with walk-forward one-step-ahead forecasting"),
        ),
        "walk_forward_window": cfg.get("walk_forward_window", artifacts.get("walk_forward_window", "expanding")),
        "rolling_window_size": cfg.get("rolling_window_size", artifacts.get("rolling_window_size")),
        "differencing_orders_exog": differencing_orders,
        "exog_lags_config": exog_lags_config,
        "sarimax_exog_cols": artifacts.get("exog_cols_sarimax") or cfg.get("sarimax_exog_cols", []),
        "sarimax_order": list(artifacts.get("best_order") or cfg.get("sarimax_order", [])),
        "sarimax_seasonal_order": list(artifacts.get("best_seasonal_order") or cfg.get("sarimax_seasonal_order", [])),
        "xgb_base_cols": artifacts.get("xgb_base_cols") or cfg.get("xgb_base_cols", []),
        "xgb_final_feature_cols": artifacts.get("final_xgb_feature_cols") or cfg.get("xgb_final_feature_cols", []),
        "residual_target": artifacts.get("residual_target") or cfg.get("residual_target", RESIDUAL_TARGET_DEFAULT),
        "target_lags": artifacts.get("target_lags") or cfg.get("target_lags", [1, 3, 6]),
        "residual_feature_cols": artifacts.get("residual_feature_cols") or cfg.get("residual_feature_cols", []),
        "threshold_red": float(cfg.get("threshold_red", artifacts.get("threshold_red", 0.05))),
        "threshold_orange": float(cfg.get("threshold_orange", artifacts.get("threshold_orange", 0.04))),
        "raw_input_columns": raw_input_columns,
        "percent_columns_decimal_in_model": percent_columns,
        "requires_future_exog_for_prediction": True,
        "requires_actual_twp90_input_for_walk_forward": True,
        "requires_raw_history_for_prediction": True,
        "prediction_input_mode": "calendar_target_twp90_with_raw_exog_and_actual_twp90_sequence",
        "prediction_input_widget": "calendar",
        "target_input_column_ui": "TWP90_Aktual_Input_%",
        "prediction_rule": "choose target TWP90 period; input raw external variables and actual TWP90 sequentially up to target period minus forecast_horizon; display all chained one-step predictions up to target",
        "prediction_display_rule": "show previous chained predictions and final selected target prediction in the same result table",
        "dashboard_max_selectable_months": int(cfg.get("dashboard_max_selectable_months", artifacts.get("dashboard_max_selectable_months", 12))),
        "model_library_sarimax": "pmdarima.ARIMA",
        "last_observed_month": str(last_observed.date()) if last_observed is not None else None,
        "next_input_month": str(next_input.date()) if next_input is not None else None,
        "next_prediction_month": str(next_prediction.date()) if next_prediction is not None else None,
        "dashboard_note": (
            "Kolom persen di UI diisi sebagai angka persen asli, misalnya inflasi 2,92% ditulis 2.92 dan TWP90 4,32% ditulis 4.32. "
            "Dashboard memilih periode TWP90 target melalui kalender. User mengisi data eksternal dan TWP90 aktual berurutan sampai bulan sebelum target agar SARIMAX dan residual history XGBoost dapat diperbarui secara one-step-ahead."
        ),
        "raw_history_status": "available" if not raw_history.empty else "missing_raw_history_csv",
    }

    with open(OUT_DIR / "preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump(enriched_config, f, ensure_ascii=False, indent=2, default=json_default)

    enriched_artifacts = dict(artifacts)
    enriched_artifacts.update(
        {
            "forecast_horizon": forecast_horizon,
            "forecasting_scheme": enriched_config["forecasting_scheme"],
            "deployment_mode": enriched_config["deployment_mode"],
            "prediction_input_widget": "calendar",
            "prediction_input_mode": enriched_config["prediction_input_mode"],
            "target_input_column_ui": enriched_config["target_input_column_ui"],
            "requires_actual_twp90_input_for_walk_forward": True,
            "raw_input_columns": raw_input_columns,
            "percent_columns_decimal_in_model": percent_columns,
            "model_library_sarimax": "pmdarima.ARIMA",
            "requires_raw_history_for_prediction": True,
        }
    )
    joblib.dump(enriched_artifacts, OUT_DIR / "hybrid_twp90_model.joblib")

    print("DONE")
    print("model_path=", MODEL_PATH)
    print("config_path=", CONFIG_PATH if CONFIG_PATH.exists() else "not found; config generated from joblib")
    print("out_dir=", OUT_DIR)
    print("raw_history_status=", enriched_config["raw_history_status"])
    print("raw_input_columns=", raw_input_columns)
    print("forecast_horizon=", forecast_horizon)
    print("last_observed_month=", enriched_config["last_observed_month"])
    print("next_input_month=", enriched_config["next_input_month"])
    print("next_prediction_month=", enriched_config["next_prediction_month"])
    if raw_history.empty:
        print("WARNING: raw_history.csv belum tersedia. Dashboard evaluasi tetap bisa membaca joblib, tetapi menu prediksi membutuhkan raw_history.csv lengkap berisi variabel eksternal historis.")


if __name__ == "__main__":
    main()
