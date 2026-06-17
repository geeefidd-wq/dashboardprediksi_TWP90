import os
import json
import copy
from html import escape
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Dashboard Prediksi TWP90 Hybrid",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
ARTIFACT_DIR = os.environ.get("MODEL_ARTIFACT_DIR", os.path.join(APP_DIR, "model_artifacts"))


def resolve_artifact_path(filename: str) -> str:
    """Cari artifact di MODEL_ARTIFACT_DIR, folder model_artifacts, lalu folder app.py.

    Logika ini tidak mengubah desain dashboard; hanya membuat deployment lebih fleksibel
    untuk paket yang menaruh joblib/json langsung satu folder dengan app.py.
    """
    candidates = []
    env_dir = os.environ.get("MODEL_ARTIFACT_DIR")
    if env_dir:
        candidates.append(os.path.join(env_dir, filename))
    candidates.extend([
        os.path.join(APP_DIR, "model_artifacts", filename),
        os.path.join(APP_DIR, filename),
    ])
    seen = set()
    unique_candidates = []
    for path in candidates:
        if path not in seen:
            unique_candidates.append(path)
            seen.add(path)
    for path in unique_candidates:
        if os.path.exists(path):
            return path
    return unique_candidates[0]


MODEL_PATH = resolve_artifact_path("hybrid_twp90_model.joblib")
CONFIG_PATH = resolve_artifact_path("preprocessing_config.json")
RAW_HISTORY_PATH = resolve_artifact_path("raw_history.csv")
DASHBOARD_HISTORY_PATH = resolve_artifact_path("dashboard_twp90_history.csv")

DATE_COL_DEFAULT = "Month"
TARGET_DEFAULT = "TWP90 (%)"
RESIDUAL_TARGET_DEFAULT = "Residual_SARIMAX_Log"
PERCENT_COL_CANDIDATES = {
    "TWP90 (%)",
    "BI-7Day-RR",
    "Inflasi",
    "Pertumbuhan Outstanding (YoY% atau MoM%)",
}

TWP90_INPUT_COL = "TWP90_Aktual_Input_%"

FRIENDLY_LABELS = {
    "Outstanding Pinjaman (miliar RP)": "Outstanding Pinjaman",
    "BI-7Day-RR": "BI-7Day-RR",
    "Inflasi": "Inflasi",
    "PDB (miliar Rp)": "PDB",
    "Pertumbuhan Outstanding (YoY% atau MoM%)": "Pertumbuhan Outstanding",
    "Indeks Keyakinan Konsumen (IKK)": "IKK",
    "Nilai Tukar Rupiah terhadap USD": "Nilai Tukar USD/IDR",
}

COLUMN_HELP = {
    "Outstanding Pinjaman (miliar RP)": "Isi sesuai satuan historis model, yaitu miliar rupiah.",
    "BI-7Day-RR": "Isi angka persen asli. Contoh: 5,75% ditulis 5.75.",
    "Inflasi": "Isi angka persen asli. Contoh: 2,92% ditulis 2.92.",
    "PDB (miliar Rp)": "Isi sesuai satuan historis model, yaitu miliar rupiah.",
    "Pertumbuhan Outstanding (YoY% atau MoM%)": "Isi angka persen asli. Contoh: 18,03% ditulis 18.03.",
    "Indeks Keyakinan Konsumen (IKK)": "Isi angka indeks IKK sesuai data/estimasi bulan tersebut.",
    "Nilai Tukar Rupiah terhadap USD": "Isi kurs rupiah terhadap USD sesuai skala historis model.",
}


def _missing_file_message(path: str) -> str:
    return (
        f"File tidak ditemukan: {path}. Pastikan artifact tersedia di folder model_artifacts "
        "atau satu folder dengan app.py."
    )


@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(_missing_file_message(MODEL_PATH))
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_raw_history(date_col, target_col):
    if not os.path.exists(RAW_HISTORY_PATH):
        return pd.DataFrame()
    df = pd.read_csv(RAW_HISTORY_PATH)
    if date_col not in df.columns:
        raise ValueError(f"Kolom tanggal '{date_col}' tidak ditemukan pada raw_history.csv.")
    df[date_col] = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp("M")
    for col in df.columns:
        if col != date_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(date_col).set_index(date_col)
    df.index.name = "Month"
    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan pada raw_history.csv.")
    return df


def history_from_joblib_artifact(artifacts, target_col):
    """Fallback tampilan/evaluasi jika raw_history.csv belum ikut disalin.

    Untuk prediksi, dashboard tetap membutuhkan raw_history.csv lengkap berisi variabel
    eksternal historis karena fitur model berasal dari differencing dan lag eksogen.
    """
    df = artifacts.get("test_results_hybrid") if isinstance(artifacts, dict) else None
    if isinstance(df, pd.DataFrame) and not df.empty and "Actual_Original" in df.columns:
        out = df[["Actual_Original"]].copy().rename(columns={"Actual_Original": target_col})
        out.index = pd.to_datetime(out.index).to_period("M").to_timestamp("M")
        out.index.name = "Month"
        return out.sort_index()
    return pd.DataFrame()


def test_predictions_from_joblib_artifact(artifacts):
    df = artifacts.get("test_results_hybrid") if isinstance(artifacts, dict) else None
    if isinstance(df, pd.DataFrame) and not df.empty:
        out = df.copy()
        if "Month" not in out.columns:
            out = out.reset_index()
        out["Month"] = pd.to_datetime(out["Month"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
        return out.sort_values("Month")
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_dashboard_history():
    if not os.path.exists(DASHBOARD_HISTORY_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DASHBOARD_HISTORY_PATH)
    df["Month"] = pd.to_datetime(df["Month"]).dt.to_period("M").dt.to_timestamp("M")
    for col in df.columns:
        if col not in ["Month", "Data_Type"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("Month")


def normalize_month_end(value):
    return pd.to_datetime(value).to_period("M").to_timestamp("M")


def month_range(start_month, horizon):
    start_period = normalize_month_end(start_month).to_period("M")
    return pd.DatetimeIndex([(start_period + i).to_timestamp("M") for i in range(int(horizon))])


def to_date(value):
    return pd.to_datetime(value).date()


def normalize_percent_to_decimal(value):
    value = float(value)
    return value / 100.0 if abs(value) > 1 else value


def to_percent_display(value):
    if pd.isna(value):
        return 0.0
    value = float(value)
    return value * 100.0 if abs(value) <= 1 else value


def fmt_pct(value, digits=2):
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}%"


def get_target_col(cfg, artifacts):
    return cfg.get("target") or artifacts.get("target") or TARGET_DEFAULT


def get_date_col(cfg):
    return cfg.get("date_col", DATE_COL_DEFAULT)


def get_exog_input_columns(cfg, raw_history, target_col, artifacts=None):
    artifact_cols = [] if artifacts is None else artifacts.get("raw_input_columns", [])
    configured_cols = cfg.get("raw_input_columns") or artifact_cols
    if configured_cols:
        return [c for c in configured_cols if c != target_col]

    diff_cols = list((cfg.get("differencing_orders_exog") or (artifacts or {}).get("differencing_orders_exog") or {}).keys())
    if diff_cols:
        return [c for c in diff_cols if c != target_col]

    if raw_history is None or raw_history.empty:
        return []
    return [c for c in raw_history.columns if c != target_col]


def get_percent_columns(cfg, exog_cols, artifacts=None):
    artifact_cols = [] if artifacts is None else artifacts.get("percent_columns_decimal_in_model", [])
    percent_cols = set(cfg.get("percent_columns_decimal_in_model", []) or artifact_cols)
    if not percent_cols:
        percent_cols = set(PERCENT_COL_CANDIDATES)
    return {c for c in exog_cols if c in percent_cols}


def dummy_covid_for_index(index):
    period_index = pd.DatetimeIndex(index).to_period("M")
    return ((period_index >= "2020-03") & (period_index <= "2021-12")).astype(int)


def classify_risk(value, orange=0.04, red=0.05):
    if value < orange:
        return "AMAN", "#16a34a", f"TWP90 masih di bawah {orange * 100:.0f}%."
    if value < red:
        return "WASPADA", "#f97316", f"TWP90 berada pada zona {orange * 100:.0f}% sampai kurang dari {red * 100:.0f}%."
    return "BAHAYA", "#dc2626", f"TWP90 melewati ambang {red * 100:.0f}%."


def get_risk_thresholds(cfg, artifacts):
    orange = float(cfg.get("threshold_orange", artifacts.get("threshold_orange", 0.04)))
    red = float(cfg.get("threshold_red", artifacts.get("threshold_red", 0.05)))
    return orange, red


def risk_status_from_percent(value_pct, orange, red):
    status, _, _ = classify_risk(float(value_pct) / 100.0, orange, red)
    return status


def make_display_name(col, percent_cols):
    label = FRIENDLY_LABELS.get(col, col)
    if col in percent_cols:
        return f"{label} (%)"
    return label


def prepare_future_input_template(raw_history, future_months, exog_cols, percent_cols, mode="zero"):
    rows = []
    for month in future_months:
        row = {"Month": month.strftime("%Y-%m")}
        for col in exog_cols:
            row[col] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def convert_editor_to_future_raw(editor_df, exog_cols, percent_cols):
    future = editor_df.copy()
    future["Month"] = pd.to_datetime(future["Month"].astype(str) + "-01", errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    if future["Month"].isna().any():
        raise ValueError("Kolom Month pada tabel input tidak valid. Gunakan format YYYY-MM.")
    if future["Month"].duplicated().any():
        raise ValueError("Terdapat bulan prediksi yang duplikat. Pastikan setiap bulan hanya muncul satu kali.")

    clean = pd.DataFrame(index=future["Month"])
    for col in exog_cols:
        if col not in future.columns:
            raise ValueError(f"Kolom input '{col}' tidak ditemukan.")
        values = pd.to_numeric(future[col], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Kolom '{col}' masih mengandung nilai kosong atau bukan angka.")
        if col in percent_cols:
            values = values.apply(normalize_percent_to_decimal)
        clean[col] = values.values
    clean.index.name = "Month"
    return clean.sort_index()


def convert_editor_to_future_target(editor_df, target_input_col=TWP90_INPUT_COL, target_col=TARGET_DEFAULT):
    """Konversi input TWP90 aktual dari UI ke skala desimal internal model.

    Di UI user mengisi angka persen asli, misalnya 4,32% ditulis 4.32.
    Di balik layar nilainya dinormalisasi menjadi 0.0432 agar konsisten dengan target model.
    """
    future = editor_df.copy()
    future["Month"] = pd.to_datetime(future["Month"].astype(str) + "-01", errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    if future["Month"].isna().any():
        raise ValueError("Kolom Month pada input TWP90 tidak valid. Gunakan format YYYY-MM.")
    if target_input_col not in future.columns:
        raise ValueError("Kolom input TWP90 aktual belum tersedia pada form prediksi.")

    values = pd.to_numeric(future[target_input_col], errors="coerce")
    if values.isna().any():
        raise ValueError("Nilai TWP90 aktual masih mengandung nilai kosong atau bukan angka.")
    values = values.apply(normalize_percent_to_decimal)
    if (values <= 0).any():
        raise ValueError("Nilai TWP90 aktual harus lebih besar dari 0.")

    out = pd.Series(values.values, index=future["Month"], name=target_col)
    out.index.name = "Month"
    return out.sort_index()


def prepare_stationary_exog_from_raw(raw_all, differencing_orders):
    raw_all = raw_all.copy().sort_index()
    raw_all.index = pd.to_datetime(raw_all.index).to_period("M").to_timestamp("M")

    for col in raw_all.columns:
        raw_all[col] = pd.to_numeric(raw_all[col], errors="coerce")

    raw_all = raw_all.ffill()

    stationary = pd.DataFrame(index=raw_all.index)
    for col, order in differencing_orders.items():
        if col not in raw_all.columns:
            continue
        order = int(order)
        if order == 0:
            stationary[col] = raw_all[col]
        elif order == 1:
            stationary[col] = raw_all[col].diff()
        elif order == 2:
            stationary[col] = raw_all[col].diff().diff()
        else:
            raise ValueError(f"Differencing order {order} belum didukung untuk kolom {col}.")
    return stationary


def build_features_from_stationary(stationary, exog_lags_config):
    full = stationary.copy().sort_index()
    full["Year"] = full.index.year
    full["Month_Num"] = full.index.month
    full["dummy_covid"] = dummy_covid_for_index(full.index)

    for col, lags in exog_lags_config.items():
        if col not in full.columns:
            continue
        for lag in lags:
            full[f"{col}_lag{int(lag)}"] = full[col].shift(int(lag))
    return full


def _extract_residual_feature_spec(residual_feature_cols, residual_target, target_lags):
    """Ambil kebutuhan fitur residual dari metadata artifact.

    Beberapa artifact menyimpan fitur seperti Residual_SARIMAX_Log_roll_std_6.
    Fungsi ini membuat dashboard mengikuti daftar fitur pada joblib/json, bukan
    mengunci fitur rolling hanya pada window tertentu.
    """
    lag_set = {int(lag) for lag in (target_lags or [])}
    roll_mean_windows = {3, 6}
    roll_std_windows = {3, 6}

    prefix = f"{residual_target}_"
    for feature in residual_feature_cols or []:
        feature = str(feature)
        if feature.startswith(f"{prefix}lag"):
            try:
                lag_set.add(int(feature.replace(f"{prefix}lag", "")))
            except ValueError:
                pass
        elif feature.startswith(f"{prefix}roll_mean_"):
            try:
                roll_mean_windows.add(int(feature.replace(f"{prefix}roll_mean_", "")))
            except ValueError:
                pass
        elif feature.startswith(f"{prefix}roll_std_"):
            try:
                roll_std_windows.add(int(feature.replace(f"{prefix}roll_std_", "")))
            except ValueError:
                pass

    return sorted(lag_set), sorted(roll_mean_windows), sorted(roll_std_windows)


def add_residual_lag_features(df, residual_target, target_lags, residual_feature_cols=None):
    df = df.copy()
    lag_list, roll_mean_windows, roll_std_windows = _extract_residual_feature_spec(
        residual_feature_cols,
        residual_target,
        target_lags,
    )

    residual_series = df[residual_target].shift(1)
    for lag in lag_list:
        df[f"{residual_target}_lag{int(lag)}"] = df[residual_target].shift(int(lag))
    for window in roll_mean_windows:
        df[f"{residual_target}_roll_mean_{int(window)}"] = residual_series.rolling(window=int(window)).mean()
    for window in roll_std_windows:
        df[f"{residual_target}_roll_std_{int(window)}"] = residual_series.rolling(window=int(window)).std()
    return df


def predict_xgb_residual_recursive(model, X_future_base, residual_history, feature_columns, residual_target, residual_feature_cols, target_lags):
    residual_history = pd.Series(residual_history).copy()
    residual_history.index = pd.to_datetime(residual_history.index).to_period("M").to_timestamp("M")
    residual_history.name = residual_target
    preds = []
    feature_rows = []

    for idx in X_future_base.index:
        temp = pd.concat([
            residual_history,
            pd.Series([np.nan], index=[idx], name=residual_target),
        ])
        temp_df = add_residual_lag_features(pd.DataFrame({residual_target: temp}), residual_target, target_lags, residual_feature_cols)
        residual_row = temp_df.loc[[idx], residual_feature_cols]
        x_row = pd.concat([X_future_base.loc[[idx]].copy(), residual_row], axis=1).reindex(columns=feature_columns)

        if x_row.isna().any().any():
            missing = x_row.columns[x_row.isna().any()].tolist()
            raise ValueError(f"Fitur residual belum lengkap untuk {idx.strftime('%Y-%m')}: {missing}")

        pred = float(model.predict(x_row)[0])
        preds.append(pred)
        feature_rows.append(x_row)
        residual_history.loc[idx] = pred

    return pd.Series(preds, index=X_future_base.index, name="Prediksi_Residual_XGB_Log"), pd.concat(feature_rows, axis=0)


def derive_xgb_base_cols_from_final(final_cols, residual_feature_cols):
    residual_set = set(residual_feature_cols)
    return [c for c in final_cols if c not in residual_set]


def get_forecast_horizon(cfg, artifacts):
    try:
        return max(1, int(cfg.get("forecast_horizon", artifacts.get("forecast_horizon", 1))))
    except Exception:
        return 1


def ensure_raw_history_has_required_exog(raw_history, differencing_orders):
    required_cols = list((differencing_orders or {}).keys())
    missing_cols = [col for col in required_cols if col not in raw_history.columns]
    if missing_cols:
        raise ValueError(
            "raw_history.csv lengkap diperlukan untuk membentuk differencing dan lag variabel eksternal. "
            "Kolom historis yang belum tersedia: " + ", ".join(missing_cols) + "."
        )


def _predict_sarimax_one_step(sarimax_model, X_row, model_library):
    """Prediksi satu langkah dari model SARIMAX/pmdarima yang sedang aktif."""
    if hasattr(sarimax_model, "predict") and model_library == "pmdarima.ARIMA":
        pred = sarimax_model.predict(n_periods=1, X=X_row)
    elif hasattr(sarimax_model, "forecast"):
        pred = sarimax_model.forecast(steps=1, exog=X_row)
    else:
        pred = sarimax_model.predict(n_periods=1, X=X_row)
    return float(np.asarray(pred, dtype=float).reshape(-1)[0])


def _update_sarimax_with_actual(sarimax_model, actual_value, X_row):
    """Update state SARIMAX dengan TWP90 aktual input user tanpa menyimpan ulang model artifact."""
    if not hasattr(sarimax_model, "update"):
        return sarimax_model
    y = np.asarray([float(actual_value)], dtype=float)
    try:
        sarimax_model.update(y, X=X_row, maxiter=0)
    except Exception:
        try:
            sarimax_model.update(y, X=X_row, maxiter=1)
        except Exception:
            sarimax_model.update(y, X=X_row)
    return sarimax_model


def _predict_xgb_residual_one_step(model, X_base_row, residual_history, feature_columns, residual_target, residual_feature_cols, target_lags):
    """Prediksi residual XGBoost satu langkah dengan residual history terbaru."""
    idx = X_base_row.index[0]
    temp = pd.concat([
        residual_history,
        pd.Series([np.nan], index=[idx], name=residual_target),
    ])
    temp_df = add_residual_lag_features(
        pd.DataFrame({residual_target: temp}),
        residual_target,
        target_lags,
        residual_feature_cols,
    )
    residual_row = temp_df.loc[[idx], residual_feature_cols] if residual_feature_cols else pd.DataFrame(index=[idx])
    x_row = pd.concat([X_base_row.copy(), residual_row], axis=1).reindex(columns=feature_columns)
    if x_row.isna().any().any():
        missing = x_row.columns[x_row.isna().any()].tolist()
        raise ValueError(f"Fitur residual belum lengkap untuk {idx.strftime('%Y-%m')}: {missing}")
    pred = float(model.predict(x_row)[0])
    return pred, x_row


def predict_hybrid_from_latest_input(raw_history, input_raw_exog, artifacts, cfg, input_raw_target=None):
    """Prediksi dashboard berbasis target TWP90 aktual input user.

    Alur:
    - User memilih periode TWP90 target yang ingin dicari.
    - User mengisi variabel eksternal dan TWP90 aktual untuk bulan-bulan sebelum target.
    - Sistem melakukan one-step-ahead secara berurutan.
    - TWP90 aktual input dipakai untuk memperbarui state SARIMAX dan residual history XGBoost.
    - Semua prediksi bulan sebelumnya sampai target akhir ditampilkan.
    """
    input_months = pd.DatetimeIndex(input_raw_exog.index).sort_values()
    if len(input_months) == 0:
        raise ValueError("Minimal harus ada satu periode input.")

    target_col = get_target_col(cfg, artifacts)
    last_observed = normalize_month_end(cfg.get("last_observed_month", raw_history.index.max()))
    required_input_month = (last_observed.to_period("M") + 1).to_timestamp("M")
    if input_months[0] != required_input_month:
        raise ValueError(
            f"Periode input harus dimulai dari bulan setelah data historis terakhir, yaitu {required_input_month.strftime('%Y-%m')}."
        )

    expected_input_months = month_range(required_input_month, len(input_months))
    if not input_months.equals(expected_input_months):
        raise ValueError("Periode input harus berurutan tanpa ada bulan yang terlewat.")

    input_target_series = None
    if input_raw_target is not None:
        input_target_series = pd.Series(input_raw_target).copy().sort_index()
        input_target_series.index = pd.to_datetime(input_target_series.index).to_period("M").to_timestamp("M")
        missing_target = [m for m in input_months if m not in input_target_series.index or pd.isna(input_target_series.loc[m])]
        if missing_target:
            raise ValueError("TWP90 aktual wajib diisi untuk seluruh periode input sebelum bulan target.")
        if (input_target_series.reindex(input_months) <= 0).any():
            raise ValueError("TWP90 aktual harus lebih besar dari 0 untuk seluruh periode input.")

    selected_input_month = input_months[-1]
    forecast_horizon = get_forecast_horizon(cfg, artifacts)
    target_month = (selected_input_month.to_period("M") + forecast_horizon).to_timestamp("M")
    first_model_forecast_month = required_input_month
    internal_horizon = target_month.to_period("M").ordinal - first_model_forecast_month.to_period("M").ordinal + 1
    forecast_months = month_range(first_model_forecast_month, internal_horizon)

    differencing_orders = artifacts.get("differencing_orders_exog") or cfg.get("differencing_orders_exog")
    exog_lags_config = artifacts.get("exog_lags_config") or cfg.get("exog_lags_config")
    if not differencing_orders or not exog_lags_config:
        raise ValueError("Config differencing atau lag eksogen tidak ditemukan pada artifact.")
    ensure_raw_history_has_required_exog(raw_history, differencing_orders)

    future_raw = input_raw_exog.copy()
    if input_target_series is not None:
        future_raw[target_col] = input_target_series.reindex(future_raw.index)
    raw_all = pd.concat([raw_history.copy(), future_raw], axis=0).sort_index()
    raw_all = raw_all[~raw_all.index.duplicated(keep="last")]

    stationary = prepare_stationary_exog_from_raw(raw_all, differencing_orders)
    missing_forecast_rows = [m for m in forecast_months if m not in stationary.index]
    if missing_forecast_rows:
        stationary = pd.concat([stationary, pd.DataFrame(index=pd.DatetimeIndex(missing_forecast_rows))], axis=0).sort_index()

    features = build_features_from_stationary(stationary, exog_lags_config)
    feature_rows = features.loc[forecast_months]

    sarimax_cols = artifacts.get("exog_cols_sarimax") or cfg.get("sarimax_exog_cols", [])
    xgb_final_cols = artifacts.get("final_xgb_feature_cols") or cfg.get("xgb_final_feature_cols", [])
    residual_target = artifacts.get("residual_target") or cfg.get("residual_target", RESIDUAL_TARGET_DEFAULT)
    residual_feature_cols = artifacts.get("residual_feature_cols") or cfg.get("residual_feature_cols", [])
    target_lags = artifacts.get("target_lags") or cfg.get("target_lags", [1, 3, 6])
    xgb_base_cols = artifacts.get("xgb_base_cols") or cfg.get("xgb_base_cols") or derive_xgb_base_cols_from_final(xgb_final_cols, residual_feature_cols)

    if not sarimax_cols:
        raise ValueError("Daftar fitur SARIMAX tidak ditemukan pada artifact/config.")
    if not xgb_base_cols or not xgb_final_cols:
        raise ValueError("Daftar fitur XGBoost tidak ditemukan pada artifact/config.")

    X_sarimax = feature_rows.reindex(columns=sarimax_cols).astype(float)
    X_xgb_base = feature_rows.reindex(columns=xgb_base_cols).astype(float)

    if X_sarimax.isna().any().any():
        missing = X_sarimax.columns[X_sarimax.isna().any()].tolist()
        raise ValueError(f"Fitur SARIMAX belum lengkap: {missing}")
    if X_xgb_base.isna().any().any():
        missing = X_xgb_base.columns[X_xgb_base.isna().any()].tolist()
        raise ValueError(f"Fitur XGBoost dasar belum lengkap: {missing}")

    sarimax_model = copy.deepcopy(artifacts["final_sarimax"])
    model_library = artifacts.get("model_library_sarimax", cfg.get("model_library_sarimax", "pmdarima.ARIMA"))
    residual_history = pd.Series(artifacts["final_residual_train_log"]).copy()
    residual_history.index = pd.to_datetime(residual_history.index).to_period("M").to_timestamp("M")
    residual_history.name = residual_target

    rows = []
    xgb_feature_rows = []
    use_log_target = artifacts.get("use_log_target", cfg.get("use_log_target", True))

    for idx in forecast_months:
        X_sarimax_row = X_sarimax.loc[[idx]]
        X_xgb_base_row = X_xgb_base.loc[[idx]]

        sarimax_pred_log = _predict_sarimax_one_step(sarimax_model, X_sarimax_row, model_library)
        xgb_resid_pred_log, xgb_row = _predict_xgb_residual_one_step(
            artifacts["final_xgb"],
            X_xgb_base_row,
            residual_history,
            xgb_final_cols,
            residual_target,
            residual_feature_cols,
            target_lags,
        )
        xgb_feature_rows.append(xgb_row)

        hybrid_log = sarimax_pred_log + xgb_resid_pred_log
        if use_log_target:
            sarimax_original = float(np.exp(sarimax_pred_log))
            hybrid_original = float(np.exp(hybrid_log))
        else:
            sarimax_original = float(sarimax_pred_log)
            hybrid_original = float(hybrid_log)

        actual_original = np.nan
        actual_log = np.nan
        actual_residual_log = np.nan
        source_note = "Target prediksi"

        if input_target_series is not None and idx in input_target_series.index:
            actual_original = float(input_target_series.loc[idx])
            actual_log = float(np.log(actual_original)) if use_log_target else actual_original
            actual_residual_log = actual_log - sarimax_pred_log
            residual_history.loc[idx] = actual_residual_log
            sarimax_model = _update_sarimax_with_actual(sarimax_model, actual_log, X_sarimax_row)
            source_note = "Aktual TWP90 input"
        else:
            residual_history.loc[idx] = xgb_resid_pred_log

        rows.append({
            "Month": idx,
            "Input_Month": selected_input_month,
            "Target_Output_Month": target_month,
            "Prediksi_SARIMAX_Original": sarimax_original,
            "Prediksi_Residual_XGB_Log": xgb_resid_pred_log,
            "Prediksi_Hybrid_Original": hybrid_original,
            "Aktual_TWP90_Input_Original": actual_original,
            "Aktual_TWP90_Input_Log": actual_log,
            "Residual_Aktual_SARIMAX_Log": actual_residual_log,
            "Sumber_TWP90": source_note,
        })

    X_xgb_final_used = pd.concat(xgb_feature_rows, axis=0) if xgb_feature_rows else pd.DataFrame()
    internal_result = pd.DataFrame(rows)
    internal_result["Prediksi_TWP90_%"] = internal_result["Prediksi_Hybrid_Original"] * 100
    internal_result["Aktual_TWP90_Input_%"] = internal_result["Aktual_TWP90_Input_Original"] * 100
    internal_result["Error_Hybrid_pp"] = internal_result["Prediksi_TWP90_%"] - internal_result["Aktual_TWP90_Input_%"]
    internal_result["Abs_Error_Hybrid_pp"] = internal_result["Error_Hybrid_pp"].abs()

    orange = cfg.get("threshold_orange", artifacts.get("threshold_orange", 0.04))
    red = cfg.get("threshold_red", artifacts.get("threshold_red", 0.05))
    internal_result[["Status", "Warna", "Keterangan"]] = internal_result["Prediksi_Hybrid_Original"].apply(
        lambda x: pd.Series(classify_risk(x, orange, red))
    )

    return internal_result, X_sarimax, X_xgb_base, X_xgb_final_used, feature_rows, raw_all, internal_result


def predict_hybrid_future(raw_history, future_raw_exog, artifacts, cfg):
    future_months = pd.DatetimeIndex(future_raw_exog.index).sort_values()
    if len(future_months) == 0:
        raise ValueError("Minimal harus ada satu bulan prediksi.")

    last_observed = normalize_month_end(cfg.get("last_observed_month", raw_history.index.max()))
    required_first = (last_observed.to_period("M") + 1).to_timestamp("M")
    if future_months[0] != required_first:
        raise ValueError(
            f"Prediksi harus dimulai dari bulan setelah data historis terakhir, yaitu {required_first.strftime('%Y-%m')}."
        )

    expected_months = month_range(required_first, len(future_months))
    if not future_months.equals(expected_months):
        raise ValueError("Bulan prediksi harus berurutan tanpa ada bulan yang terlewat.")

    raw_all = pd.concat([raw_history.copy(), future_raw_exog.copy()], axis=0).sort_index()
    raw_all = raw_all[~raw_all.index.duplicated(keep="last")]

    differencing_orders = artifacts.get("differencing_orders_exog") or cfg.get("differencing_orders_exog")
    exog_lags_config = artifacts.get("exog_lags_config") or cfg.get("exog_lags_config")
    if not differencing_orders or not exog_lags_config:
        raise ValueError("Config differencing atau lag eksogen tidak ditemukan pada artifact.")

    stationary = prepare_stationary_exog_from_raw(raw_all, differencing_orders)
    features = build_features_from_stationary(stationary, exog_lags_config)
    feature_rows = features.loc[future_months]

    sarimax_cols = artifacts.get("exog_cols_sarimax") or cfg.get("sarimax_exog_cols", [])
    xgb_final_cols = artifacts.get("final_xgb_feature_cols") or cfg.get("xgb_final_feature_cols", [])
    residual_target = artifacts.get("residual_target") or cfg.get("residual_target", RESIDUAL_TARGET_DEFAULT)
    residual_feature_cols = artifacts.get("residual_feature_cols") or cfg.get("residual_feature_cols", [])
    target_lags = artifacts.get("target_lags") or cfg.get("target_lags", [1, 3, 6])
    xgb_base_cols = artifacts.get("xgb_base_cols") or artifacts.get("xgb_base_cols") or derive_xgb_base_cols_from_final(xgb_final_cols, residual_feature_cols)

    if not sarimax_cols:
        raise ValueError("Daftar fitur SARIMAX tidak ditemukan pada artifact/config.")
    if not xgb_base_cols or not xgb_final_cols:
        raise ValueError("Daftar fitur XGBoost tidak ditemukan pada artifact/config.")

    X_sarimax = feature_rows.reindex(columns=sarimax_cols).astype(float)
    X_xgb_base = feature_rows.reindex(columns=xgb_base_cols).astype(float)

    if X_sarimax.isna().any().any():
        missing = X_sarimax.columns[X_sarimax.isna().any()].tolist()
        raise ValueError(f"Fitur SARIMAX belum lengkap: {missing}")
    if X_xgb_base.isna().any().any():
        missing = X_xgb_base.columns[X_xgb_base.isna().any()].tolist()
        raise ValueError(f"Fitur XGBoost dasar belum lengkap: {missing}")

    sarimax_model = artifacts["final_sarimax"]
    horizon = len(future_months)
    model_library = artifacts.get("model_library_sarimax", cfg.get("model_library_sarimax", "pmdarima.ARIMA"))
    if hasattr(sarimax_model, "predict") and model_library == "pmdarima.ARIMA":
        sarimax_pred = sarimax_model.predict(n_periods=horizon, X=X_sarimax)
    elif hasattr(sarimax_model, "forecast"):
        sarimax_pred = sarimax_model.forecast(steps=horizon, exog=X_sarimax)
    else:
        sarimax_pred = sarimax_model.predict(n_periods=horizon, X=X_sarimax)

    sarimax_pred_log = pd.Series(np.asarray(sarimax_pred, dtype=float).reshape(-1), index=future_months, name="Prediksi_SARIMAX_Log")
    xgb_resid_log, X_xgb_final_used = predict_xgb_residual_recursive(
        artifacts["final_xgb"],
        X_xgb_base,
        artifacts["final_residual_train_log"],
        xgb_final_cols,
        residual_target,
        residual_feature_cols,
        target_lags,
    )

    hybrid_log = (sarimax_pred_log + xgb_resid_log).rename("Prediksi_Hybrid_Log")
    if artifacts.get("use_log_target", cfg.get("use_log_target", True)):
        hybrid_original = np.exp(hybrid_log)
        sarimax_original = np.exp(sarimax_pred_log)
    else:
        hybrid_original = hybrid_log
        sarimax_original = sarimax_pred_log

    result = pd.DataFrame({
        "Month": future_months,
        "Prediksi_SARIMAX_Original": sarimax_original.values,
        "Prediksi_Residual_XGB_Log": xgb_resid_log.values,
        "Prediksi_Hybrid_Original": hybrid_original.values,
    })
    result["Prediksi_TWP90_%"] = result["Prediksi_Hybrid_Original"] * 100
    orange = cfg.get("threshold_orange", artifacts.get("threshold_orange", 0.04))
    red = cfg.get("threshold_red", artifacts.get("threshold_red", 0.05))
    result[["Status", "Warna", "Keterangan"]] = result["Prediksi_Hybrid_Original"].apply(
        lambda x: pd.Series(classify_risk(x, orange, red))
    )

    return result, X_sarimax, X_xgb_base, X_xgb_final_used, feature_rows, raw_all


TEST_PREDICTIONS_PATH = resolve_artifact_path("test_predictions_hybrid.csv")


@st.cache_data(show_spinner=False)
def load_test_predictions():
    if not os.path.exists(TEST_PREDICTIONS_PATH):
        return pd.DataFrame()
    df = pd.read_csv(TEST_PREDICTIONS_PATH)
    if "Month" in df.columns:
        df["Month"] = pd.to_datetime(df["Month"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    for col in df.columns:
        if col != "Month":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Month" in df.columns:
        df = df.sort_values("Month")
    return df


def find_first_existing(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def prepare_evaluation_frame(test_predictions, dashboard_history, target_col):
    candidates = []
    if test_predictions is not None and not test_predictions.empty:
        candidates.append(("test_predictions_hybrid.csv", test_predictions.copy()))
    if dashboard_history is not None and not dashboard_history.empty:
        candidates.append(("dashboard_twp90_history.csv", dashboard_history.copy()))

    actual_candidates = [
        "Actual_Original",
        "Actual",
        "Actual_TWP90_Original",
        "TWP90_Actual",
        target_col,
        "TWP90 (%)",
    ]
    pred_candidates = [
        "Prediksi_Hybrid_Original",
        "Hybrid_Pred_Original",
        "Prediksi_Hybrid",
        "Predicted_Original",
        "Prediction",
        "Prediksi_TWP90_Original",
    ]

    for source_name, df in candidates:
        actual_col = find_first_existing(df.columns, actual_candidates)
        pred_col = find_first_existing(df.columns, pred_candidates)
        if actual_col and pred_col:
            out = df[[c for c in ["Month", actual_col, pred_col] if c in df.columns]].copy()
            if "Month" not in out.columns:
                out["Month"] = pd.RangeIndex(1, len(out) + 1).astype(str)
            out = out.rename(columns={actual_col: "Actual", pred_col: "Predicted"})
            out["Actual"] = pd.to_numeric(out["Actual"], errors="coerce")
            out["Predicted"] = pd.to_numeric(out["Predicted"], errors="coerce")
            out = out.dropna(subset=["Actual", "Predicted"])
            if not out.empty:
                out["Source"] = source_name
                return out
    return pd.DataFrame()


def evaluation_metrics(eval_df):
    if eval_df is None or eval_df.empty:
        return {}, pd.DataFrame(), 100.0

    df = eval_df.copy()
    max_value = max(df["Actual"].abs().max(), df["Predicted"].abs().max())
    scale = 100.0 if pd.notna(max_value) and max_value <= 1.5 else 1.0

    df["Actual_%"] = df["Actual"] * scale
    df["Predicted_%"] = df["Predicted"] * scale
    df["Error_pp"] = df["Predicted_%"] - df["Actual_%"]
    df["Abs_Error_pp"] = df["Error_pp"].abs()
    nonzero = df["Actual_%"].abs() > 1e-12

    mae = float(df["Abs_Error_pp"].mean())
    rmse = float(np.sqrt((df["Error_pp"] ** 2).mean()))
    mape = float((df.loc[nonzero, "Abs_Error_pp"] / df.loc[nonzero, "Actual_%"].abs()).mean() * 100) if nonzero.any() else np.nan
    bias = float(df["Error_pp"].mean())
    sse = float(((df["Actual_%"] - df["Predicted_%"]) ** 2).sum())
    sst = float(((df["Actual_%"] - df["Actual_%"].mean()) ** 2).sum())
    r2 = float(1 - sse / sst) if sst > 0 else np.nan

    return {
        "MAE_pp": mae,
        "RMSE_pp": rmse,
        "MAPE_%": mape,
        "Bias_pp": bias,
        "R2": r2,
        "N": int(len(df)),
        "Source": str(df["Source"].iloc[0]) if "Source" in df.columns else "-",
    }, df, scale


def prepare_prediction_input_evaluation_frame(payload):
    """Membentuk data evaluasi dari hasil prediksi terbaru yang dihitung user.

    Evaluasi hanya dihitung untuk baris yang memiliki dua nilai sekaligus:
    TWP90 aktual input dan prediksi hybrid. Baris target yang belum punya TWP90
    aktual otomatis tidak dihitung agar metrik tidak bias.
    """
    if not isinstance(payload, dict) or "result" not in payload:
        return pd.DataFrame()

    result = payload.get("result")
    if result is None or not isinstance(result, pd.DataFrame) or result.empty:
        return pd.DataFrame()

    required_cols = {"Month", "Aktual_TWP90_Input_%", "Prediksi_TWP90_%"}
    if not required_cols.issubset(set(result.columns)):
        return pd.DataFrame()

    out = result[["Month", "Aktual_TWP90_Input_%", "Prediksi_TWP90_%"]].copy()
    out["Actual"] = pd.to_numeric(out["Aktual_TWP90_Input_%"], errors="coerce")
    out["Predicted"] = pd.to_numeric(out["Prediksi_TWP90_%"], errors="coerce")
    out = out.dropna(subset=["Actual", "Predicted"])

    if out.empty:
        return pd.DataFrame()

    out = out[["Month", "Actual", "Predicted"]].copy()
    out["Source"] = "Hasil prediksi input user"
    return out


def get_active_evaluation_dataset(default_eval_raw):
    """Prioritaskan evaluasi dari prediksi user, lalu fallback ke file evaluasi historis."""
    user_eval_raw = prepare_prediction_input_evaluation_frame(st.session_state.get("prediction_payload"))
    if not user_eval_raw.empty:
        summary, detail, scale = evaluation_metrics(user_eval_raw)
        return summary, detail, scale, "user_prediction"

    summary, detail, scale = evaluation_metrics(default_eval_raw)
    return summary, detail, scale, "artifact_history"


def value_card(label, value, note="", accent="#1d4ed8"):
    return f"""
    <div class="value-card" style="border-top:4px solid {accent};">
        <div class="metric-label">{label}</div>
        <div class="metric-number">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


def clean_undefined_strings(df):
    """Menghapus nilai teks undefined/null dengan menggantinya menjadi string kosong."""
    out = df.copy()
    text_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        out[col] = out[col].replace({
            "undefined": "",
            "Undefined": "",
            "UNDEFINED": "",
            "null": "",
            "None": "",
            "nan": "",
            "-": ""
        }).fillna("")
    return out


def _html_text(value):
    if pd.isna(value):
        return ""
    return escape(str(value))


def _progress_html(value, max_value, suffix="%"):
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value):
        value = 0.0
    max_value = max(float(max_value), 1.0)
    width = max(0.0, min(100.0, (float(value) / max_value) * 100.0))
    suffix = escape(str(suffix))
    return f"""
    <div class="progress-cell">
        <div class="progress-meta"><span>{float(value):.3f}</span><span>{suffix}</span></div>
        <div class="progress-track"><div class="progress-fill" style="width:{width:.2f}%;"></div></div>
    </div>
    """


def _number_html(value, digits=3, suffix=""):
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}{escape(str(suffix))}"


def _status_badge_html(status):
    status_text = _html_text(status).upper()
    css_class = "status-aman"
    if status_text == "WASPADA":
        css_class = "status-waspada"
    elif status_text == "BAHAYA":
        css_class = "status-bahaya"
    return f'<span class="status-badge {css_class}">{status_text}</span>'


def render_modern_blue_table(
    table,
    progress_cols=None,
    delta_cols=None,
    status_cols=None,
    numeric_cols=None,
    suffix_map=None,
    height=460,
):
    """Render tabel custom bertema biru agar semua menu konsisten dengan tabel dashboard."""
    table = clean_undefined_strings(table.copy())
    progress_cols = list(progress_cols or [])
    delta_cols = list(delta_cols or [])
    status_cols = list(status_cols or ["Status Risiko", "Status"])
    suffix_map = suffix_map or {}

    if numeric_cols is None:
        numeric_cols = []
        for col in table.columns:
            converted = pd.to_numeric(table[col], errors="coerce")
            if converted.notna().any() and col not in progress_cols and col not in delta_cols:
                numeric_cols.append(col)

    max_values = {}
    for col in progress_cols:
        if col in table.columns:
            numeric_values = pd.to_numeric(table[col], errors="coerce")
            if numeric_values.notna().any():
                max_values[col] = max(1.0, float(numeric_values.max()) + 1)
            else:
                max_values[col] = 1.0

    header_html = "".join(f"<th>{escape(str(col))}</th>" for col in table.columns)
    body_rows = []
    for _, row in table.iterrows():
        cells = []
        for col in table.columns:
            value = row[col]
            if col in progress_cols:
                cells.append(f"<td>{_progress_html(value, max_values[col], suffix_map.get(col, '%'))}</td>")
            elif col in delta_cols:
                delta = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                cls = "delta-pos" if pd.notna(delta) and float(delta) >= 0 else "delta-neg"
                display = "" if pd.isna(delta) else f"{float(delta):.3f}"
                cells.append(f'<td><span class="{cls}">{display}</span></td>')
            elif col in status_cols:
                cells.append(f"<td>{_status_badge_html(value)}</td>")
            elif col in numeric_cols:
                cells.append(f"<td>{_number_html(value, suffix=suffix_map.get(col, ''))}</td>")
            else:
                cells.append(f"<td>{_html_text(value)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = f"""
    <div class="blue-table-shell">
        <div class="blue-table-scroll" style="max-height:{int(height)}px;">
            <table class="blue-modern-table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{''.join(body_rows)}</tbody>
            </table>
        </div>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def render_modern_history_table(table):
    progress_cols = [c for c in ["TWP90 Aktual (%)", "Prediksi Hybrid (%)", "Prediksi TWP90 (%)"] if c in table.columns]
    delta_cols = [c for c in ["Selisih (pp)"] if c in table.columns]
    render_modern_blue_table(
        table,
        progress_cols=progress_cols,
        delta_cols=delta_cols,
        status_cols=["Status Risiko"],
        suffix_map={c: "%" for c in progress_cols},
        height=460,
    )


def rerun_dashboard():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def input_step_for_column(col, percent_cols):
    if col in percent_cols:
        return 0.01
    label = FRIENDLY_LABELS.get(col, col).lower()
    if "nilai tukar" in label or "usd" in label:
        return 100.0
    if "ikk" in label or "indeks" in label:
        return 1.0
    if "pdb" in label or "outstanding" in label:
        return 1000.0
    return 1.0


def validate_required_prediction_inputs(input_df, exog_cols, percent_cols, target_input_col=None):
    """Validasi form prediksi: seluruh input wajib diisi dan nilai 0 dianggap belum diisi."""
    if input_df is None or input_df.empty:
        raise ValueError("Hasil prediksi tidak ditemukan. Form input wajib diisi terlebih dahulu.")

    required_cols = list(exog_cols)
    if target_input_col:
        required_cols = [target_input_col] + required_cols

    missing_columns = [col for col in required_cols if col not in input_df.columns]
    if missing_columns:
        shown_missing = ["TWP90 aktual (%)" if col == target_input_col else make_display_name(col, percent_cols) for col in missing_columns]
        raise ValueError(
            "Hasil prediksi tidak ditemukan. Kolom input belum lengkap: "
            + ", ".join(shown_missing)
        )

    numeric_values = input_df[required_cols].apply(pd.to_numeric, errors="coerce")
    invalid_mask = numeric_values.isna() | (numeric_values.abs() <= 1e-12)

    if invalid_mask.any().any():
        detail_rows = []
        for idx, row in invalid_mask.iterrows():
            invalid_cols = [col for col in required_cols if bool(row.get(col, False))]
            if invalid_cols:
                period = str(input_df.loc[idx, "Month"]) if "Month" in input_df.columns else f"baris {idx + 1}"
                shown_cols = [
                    "TWP90 aktual (%)" if col == target_input_col else make_display_name(col, percent_cols)
                    for col in invalid_cols[:3]
                ]
                extra = "" if len(invalid_cols) <= 3 else f" dan {len(invalid_cols) - 3} kolom lain"
                detail_rows.append(f"{period}: {', '.join(shown_cols)}{extra}")

        detail_text = "; ".join(detail_rows[:5])
        if len(detail_rows) > 5:
            detail_text += f"; dan {len(detail_rows) - 5} periode lain"

        raise ValueError(
            "Hasil prediksi tidak ditemukan. Seluruh form input wajib diisi dan nilai 0 dianggap belum diisi. "
            f"Periksa kembali input berikut: {detail_text}."
        )


def format_feature_label(feature_name):
    """Membuat nama fitur teknis menjadi lebih mudah dibaca pada grafik evaluasi."""
    feature_name = str(feature_name)
    replacements = {
        "Residual_SARIMAX_Log": "Residual SARIMAX log",
        "BI-7Day-RR": "BI-7Day-RR",
        "Indeks Keyakinan Konsumen (IKK)": "IKK",
        "PDB (miliar Rp)": "PDB",
        "Outstanding Pinjaman (miliar RP)": "Outstanding Pinjaman",
        "Pertumbuhan Outstanding (YoY% atau MoM%)": "Pertumbuhan Outstanding",
        "Nilai Tukar Rupiah terhadap USD": "Nilai Tukar USD/IDR",
        "dummy_covid": "Dummy Covid",
        "Month_Num": "Bulan",
        "Year": "Tahun",
        "_lag": " lag ",
        "_roll_mean_": " rolling mean ",
        "_roll_std_": " rolling std ",
        "_": " ",
    }
    label = feature_name
    for old, new in replacements.items():
        label = label.replace(old, new)
    return " ".join(label.split())


def get_hybrid_feature_importance(artifacts, cfg, top_n=20):
    """Mengambil feature importance dari komponen XGBoost residual pada model hybrid."""
    model = artifacts.get("final_xgb")
    feature_cols = artifacts.get("final_xgb_feature_cols") or cfg.get("xgb_final_feature_cols") or []
    if model is None:
        return pd.DataFrame()

    importance_values = None
    importance_type = "gain"

    if hasattr(model, "feature_importances_"):
        try:
            importance_values = np.asarray(model.feature_importances_, dtype=float)
            importance_type = "feature_importances_"
        except Exception:
            importance_values = None

    if importance_values is not None and len(feature_cols) == len(importance_values):
        df = pd.DataFrame({"Feature": feature_cols, "Importance": importance_values})
    else:
        try:
            booster = model.get_booster()
            score = booster.get_score(importance_type="gain")
            if not score:
                score = booster.get_score(importance_type="weight")
                importance_type = "weight"
            rows = []
            for key, value in score.items():
                if key.startswith("f") and key[1:].isdigit() and feature_cols:
                    idx = int(key[1:])
                    feature = feature_cols[idx] if idx < len(feature_cols) else key
                else:
                    feature = key
                rows.append({"Feature": feature, "Importance": float(value)})
            df = pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df["Importance"] = pd.to_numeric(df["Importance"], errors="coerce").fillna(0.0)
    df = df[df["Importance"] > 0].copy()
    if df.empty:
        return pd.DataFrame()

    total = df["Importance"].sum()
    df["Importance_%"] = (df["Importance"] / total * 100.0) if total > 0 else 0.0
    df["Feature_Label"] = df["Feature"].apply(format_feature_label)
    df["Importance_Type"] = importance_type
    return df.sort_values("Importance_%", ascending=False).head(int(top_n)).sort_values("Importance_%", ascending=True)


def show_hybrid_feature_importance():
    importance_df = get_hybrid_feature_importance(artifacts, cfg, top_n=20)
    st.markdown('<div class="section-title">Feature Importance Model Hybrid</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-help">Grafik ini membaca feature importance dari komponen <b>XGBoost residual</b> pada model hybrid SARIMAX + XGBoost. Nilainya dinormalisasi menjadi persentase agar lebih mudah dibandingkan.</div>',
        unsafe_allow_html=True,
    )

    if importance_df.empty:
        st.markdown(
            """
            <div class="info-box">
                Feature importance belum dapat ditampilkan karena metadata fitur XGBoost atau atribut importance tidak tersedia pada artifact model.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    fig_importance = go.Figure(go.Bar(
        x=importance_df["Importance_%"],
        y=importance_df["Feature_Label"],
        orientation="h",
        text=[f"{v:.2f}%" for v in importance_df["Importance_%"]],
        textposition="outside",
        marker=dict(color=importance_df["Importance_%"], colorscale="Blues", showscale=False),
        hovertemplate="%{y}<br>Importance: %{x:.2f}%<extra></extra>",
    ))
    fig_importance.update_layout(
        height=max(430, int(len(importance_df) * 28 + 180)),
        title="Feature Importance Hybrid - XGBoost Residual",
        plot_bgcolor="rgba(240,247,255,1)",
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#0f2a5f"),
        xaxis_title="Importance (%)",
        yaxis_title="Fitur",
        margin=dict(l=20, r=70, t=70, b=30),
    )
    fig_importance.update_xaxes(range=[0, max(importance_df["Importance_%"].max() * 1.18, 1)])
    st.plotly_chart(fig_importance, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    # Tabel feature importance dihapus sesuai permintaan; evaluasi hanya menampilkan grafik dan caption.


try:
    cfg = load_config()
    artifacts = load_artifacts()
    target_col = get_target_col(cfg, artifacts)
    date_col = get_date_col(cfg)
    raw_history = load_raw_history(date_col, target_col)
    if raw_history.empty:
        raw_history = history_from_joblib_artifact(artifacts, target_col)
    if raw_history.empty:
        raise FileNotFoundError(_missing_file_message(RAW_HISTORY_PATH))
    history = load_dashboard_history()
    test_predictions = load_test_predictions()
    if test_predictions.empty:
        test_predictions = test_predictions_from_joblib_artifact(artifacts)
    if history.empty and not test_predictions.empty:
        history = test_predictions.copy()
        if "Actual_Original" in history.columns and "Data_Type" not in history.columns:
            history["Data_Type"] = "Test"
    eval_raw = prepare_evaluation_frame(test_predictions, history, target_col)
    eval_summary, eval_detail, eval_scale = evaluation_metrics(eval_raw)
    exog_cols = get_exog_input_columns(cfg, raw_history, target_col, artifacts)
    percent_cols = get_percent_columns(cfg, exog_cols, artifacts)
except Exception as init_error:
    st.error(f"Dashboard gagal memuat artifact: {init_error}")
    st.stop()

last_observed = normalize_month_end(cfg.get("last_observed_month", raw_history.index.max()))
next_month = (last_observed.to_period("M") + 1).to_timestamp("M")
DASHBOARD_FORECAST_HORIZON = get_forecast_horizon(cfg, artifacts)
target_output_month = (next_month.to_period("M") + DASHBOARD_FORECAST_HORIZON).to_timestamp("M")
try:
    MAX_FORECAST_MONTHS = max(1, int(cfg.get("dashboard_max_selectable_months", artifacts.get("dashboard_max_selectable_months", 12))))
except Exception:
    MAX_FORECAST_MONTHS = 12
available_input_months = month_range(next_month, MAX_FORECAST_MONTHS)
available_target_months = pd.DatetimeIndex([
    (month.to_period("M") + DASHBOARD_FORECAST_HORIZON).to_timestamp("M")
    for month in available_input_months
])

last_actual = raw_history[target_col].dropna().iloc[-1]
last_actual_pct = to_percent_display(last_actual)

latest_history_pred_pct = None
latest_history_pred_month = None
if not history.empty and "Prediksi_Hybrid_Original" in history.columns:
    pred_hist = history.dropna(subset=["Prediksi_Hybrid_Original"]).copy()
    if not pred_hist.empty:
        pred_hist = pred_hist.sort_values("Month")
        latest_history_pred_pct = to_percent_display(pred_hist["Prediksi_Hybrid_Original"].iloc[-1])
        latest_history_pred_month = pred_hist["Month"].iloc[-1]

active_eval_summary, active_eval_detail, active_eval_scale, active_eval_source = get_active_evaluation_dataset(eval_raw)
mape_value = active_eval_summary.get("MAPE_%") if active_eval_summary else np.nan
mape_text = "-" if pd.isna(mape_value) else f"{mape_value:.2f}%"
RISK_ORANGE, RISK_RED = get_risk_thresholds(cfg, artifacts)
RISK_ORANGE_PCT = RISK_ORANGE * 100
RISK_RED_PCT = RISK_RED * 100

st.markdown(
    """
<style>
:root{
    --blue-950:#071a3f;
    --blue-900:#0f2a5f;
    --blue-800:#1e3a8a;
    --blue-700:#1d4ed8;
    --blue-600:#2563eb;
    --blue-500:#3b82f6;
    --blue-100:#dbeafe;
    --blue-050:#eff6ff;
    --cyan-400:#38bdf8;
    --slate-500:#64748b;
    --slate-900:#0f172a;
}
.stApp {
    background:
        radial-gradient(circle at 12% 12%, rgba(56,189,248,.16) 0, rgba(56,189,248,0) 34%),
        linear-gradient(180deg,#f4f9ff 0%, #eaf4ff 100%);
}
.block-container {padding-top:2.15rem; padding-bottom:2.75rem; max-width:1320px;}

/* Sidebar sesuai referensi: compact, modern, tanpa emoji */
[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg,#071a3f 0%, #0d2557 42%, #153982 72%, #1d4ed8 100%) !important;
    border-right:1px solid rgba(255,255,255,.10);
    box-shadow: 12px 0 34px rgba(15,42,95,.24);
}
[data-testid="stSidebar"]::before {display:none !important;}
[data-testid="stSidebar"] * {color:#f8fafc !important;}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding:1.45rem 1.35rem 1.5rem 1.35rem;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.72rem;}

.sidebar-brand {
    min-height:158px;
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    text-align:center;
    padding:1.25rem .72rem 1.05rem .72rem;
    margin:.15rem 0 1.15rem 0;
}
.sidebar-title {
    width:100%;
    font-size:32px; /* Judul sidebar diperbesar sesuai permintaan */
    font-weight:1000;
    letter-spacing:-.045em;
    line-height:1.08;
    color:#ffffff;
}
.sidebar-subtitle {
    font-size:15.5px; /* Subjudul hybrid juga diperbesar */
    color:#38bdf8 !important;
    opacity:1;
    margin-top:13px;
    font-weight:900;
    line-height:1.28;
    text-align:center;
}
.sidebar-divider {
    width:100%;
    height:1px;
    margin:20px 0 0 0;
    background:linear-gradient(90deg,rgba(255,255,255,0),rgba(255,255,255,.22),rgba(255,255,255,0));
}

/* Tombol navigasi sidebar dibuat seperti kartu pada contoh */
[data-testid="stSidebar"] div.stButton > button {
    width:100%;
    min-height:54px !important;
    justify-content:center !important;
    text-align:center !important;
    border-radius:12px !important;
    border:1px solid rgba(255,255,255,.12) !important;
    background:rgba(255,255,255,.055) !important;
    color:#f8fafc !important;
    box-shadow:none !important;
    margin:5px 0 12px 0;
    padding:0 16px !important;
    font-size:15px !important;
    font-weight:900 !important;
    letter-spacing:-.01em;
    transition:all .18s ease-in-out;
}
[data-testid="stSidebar"] div.stButton > button:hover {
    background:linear-gradient(135deg,rgba(56,189,248,.22),rgba(255,255,255,.08)) !important;
    border-color:rgba(255,255,255,.28) !important;
    transform:translateY(-1px);
    color:#ffffff !important;
}

.hero {
    position:relative; overflow:hidden; color:white; border-radius:32px; padding:42px 38px 38px 38px;
    min-height:156px;
    background:linear-gradient(135deg,#061536 0%, #0f2a5f 42%, #1d4ed8 78%, #38bdf8 100%);
    box-shadow:0 24px 64px rgba(15,42,95,.15); margin:8px 0 22px 0;
}
.hero:before {content:""; position:absolute; width:340px; height:340px; border-radius:999px; right:-120px; top:-120px; background:rgba(255,255,255,.14);}
.hero:after {content:""; position:absolute; width:220px; height:220px; border-radius:999px; right:160px; bottom:-150px; background:rgba(125,211,252,.16);}
.hero-title {position:relative; font-size:42px; line-height:1.18; font-weight:1000; margin:0 0 10px 0; letter-spacing:-.04em; padding-top:2px; text-transform:none;}
.badge-row {position:relative; display:flex; gap:10px; flex-wrap:wrap; margin-top:18px;}
.badge {background:rgba(255,255,255,.15); border:1px solid rgba(255,255,255,.28); color:white; border-radius:999px; padding:8px 13px; font-size:12px; font-weight:800; backdrop-filter:blur(10px);}

.panel, .value-card, .result-panel, .input-panel {
    background:rgba(255,255,255,.98); border:1px solid #d7e7ff; border-radius:24px;
    box-shadow:0 14px 34px rgba(30,64,175,.06); padding:20px 22px;
}
.input-panel {padding:18px 20px; margin-bottom:14px;}
.value-card {min-height:118px;}
.metric-label {font-size:11px; color:#64748b; font-weight:900; text-transform:uppercase; letter-spacing:.075em;}
.metric-number {font-size:30px; color:#0f2a5f; font-weight:950; margin-top:6px; letter-spacing:-.02em;}
.metric-note {font-size:12.5px; color:#64748b; line-height:1.45; margin-top:4px;}
.section-title {font-size:23px; font-weight:1000; color:#0f2a5f; margin-top:8px; margin-bottom:4px; letter-spacing:-.025em;}
.section-help {font-size:13.5px; color:#64748b; margin-bottom:14px; line-height:1.6;}
.info-box {background:#eff6ff; border:1px solid #bfdbfe; border-radius:18px; padding:14px 16px; color:#1e3a8a; font-size:13px; line-height:1.58;}
.warn-box {background:#fff7ed; border:1px solid #fed7aa; border-radius:18px; padding:14px 16px; color:#9a3412; font-size:13px; line-height:1.58;}
.status-pill {border-radius:999px; padding:6px 12px; font-size:12px; color:white; font-weight:900; display:inline-block; letter-spacing:.03em;}
.result-panel {padding:24px 26px; border-left:7px solid #1d4ed8;}
.result-title {font-size:17px; font-weight:950; color:#0f2a5f; margin-bottom:8px;}
.result-main {font-size:46px; font-weight:1000; color:#0f2a5f; letter-spacing:-.04em; line-height:1; margin:8px 0 10px 0;}
.result-sub {font-size:13px; color:#64748b; line-height:1.55;}
.eyebrow {font-size:12px; font-weight:900; color:#1d4ed8; text-transform:uppercase; letter-spacing:.09em;}
.component-grid {display:grid; grid-template-columns:1fr; gap:10px; margin-top:12px;}
.component-item {background:#f8fbff; border:1px solid #dbeafe; border-radius:16px; padding:12px 14px;}
.component-label {font-size:11px; color:#64748b; font-weight:900; text-transform:uppercase; letter-spacing:.07em;}
.component-value {font-size:19px; color:#0f2a5f; font-weight:950; margin-top:4px;}
.modern-table-title {font-size:15px; font-weight:950; color:#0f2a5f; margin:14px 0 8px;}
.eval-table-title {font-size:24px; font-weight:1000; color:#0f2a5f; margin:26px 0 12px; letter-spacing:-.025em;}
.feature-importance-spacer {height:46px;}
.table-note {font-size:12.5px; color:#64748b; line-height:1.55; margin:-2px 0 12px 0;}
div.stButton > button, div.stDownloadButton > button, div.stFormSubmitButton > button {
    border-radius:16px !important; border:0 !important; min-height:48px !important; font-weight:900 !important;
    background:linear-gradient(135deg,#0f2a5f 0%,#1d4ed8 58%,#0ea5e9 100%) !important; color:white !important;
    box-shadow:0 14px 28px rgba(15,42,95,.24) !important;
}
div.stButton > button:hover, div.stDownloadButton > button:hover, div.stFormSubmitButton > button:hover {filter:brightness(1.05); transform:translateY(-1px);}

/* Form Input & Element Interaktif */
[data-testid="stNumberInput"] input {
    border-radius:14px !important; border:1px solid #bfdbfe !important; background:#ffffff !important;
    color:#0f2a5f !important; font-weight:800 !important;
}
[data-testid="stDateInput"] div[data-baseweb="input"] > div,
[data-testid="stDateInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stTextInput"] input {
    border-radius:14px !important;
    border-color:#bfdbfe !important;
    background:#ffffff !important;
    background-color:#ffffff !important;
    color:#0f2a5f !important;
    font-weight:800 !important;
}
[data-testid="stDateInput"] div[data-baseweb="input"],
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background:#ffffff !important;
    background-color:#ffffff !important;
    border-radius:14px !important;
}
[data-testid="stDateInput"] svg,
[data-testid="stSelectbox"] svg {
    color:#0f2a5f !important;
    fill:#0f2a5f !important;
}
.stTabs [data-baseweb="tab-list"] {gap:8px;}
.stTabs [data-baseweb="tab"] {
    border-radius:999px; background:#e0ecff; color:#1e3a8a; font-weight:900; padding:8px 14px;
}
.stTabs [aria-selected="true"] {background:linear-gradient(135deg,#1d4ed8,#38bdf8) !important; color:white !important;}

/* Hapus teks judul plotly undefined dan modebar agar grafik bersih */
[data-testid="stPlotlyChart"] .gtitle {display:none !important;}
[data-testid="stPlotlyChart"] .modebar-container {display:none !important;}

/* Desain Tabel Modern Tema Biru */
[data-testid="stDataFrame"] {
    border: 1px solid #cce0ff !important; border-radius: 16px !important; overflow: hidden !important;
    box-shadow: 0 8px 24px rgba(30, 64, 175, 0.08) !important; background: white !important;
}
[data-testid="stDataFrame"] div[role="columnheader"] {
    background: linear-gradient(135deg,#0f2a5f,#1d4ed8) !important; color: #ffffff !important; font-weight: 900 !important;
}
[data-testid="stDataFrame"] div[role="gridcell"] {
    font-size: 13.5px !important; color: #0f2a5f !important;
}

.blue-table-shell{
    border:1px solid #c7ddff; border-radius:22px; overflow:hidden; background:#ffffff;
    box-shadow:0 16px 38px rgba(30,64,175,.10); margin-top:8px;
}
.blue-table-scroll{max-height:460px; overflow:auto;}
.blue-modern-table{width:100%; border-collapse:separate; border-spacing:0; font-size:13.5px; color:#0f2a5f;}
.blue-modern-table thead th{
    position:sticky; top:0; z-index:2; text-align:left; padding:14px 16px;
    background:linear-gradient(135deg,#0f2a5f 0%,#1d4ed8 64%,#38bdf8 100%);
    color:#ffffff; font-size:12px; text-transform:uppercase; letter-spacing:.07em; font-weight:950;
    border-bottom:1px solid rgba(255,255,255,.22);
}
.blue-modern-table tbody tr:nth-child(odd){background:#ffffff;}
.blue-modern-table tbody tr:nth-child(even){background:#f6faff;}
.blue-modern-table tbody tr:hover{background:#eaf4ff;}
.blue-modern-table td{padding:12px 16px; border-bottom:1px solid #e2efff; vertical-align:middle; font-weight:700;}
.progress-cell{min-width:180px;}
.progress-meta{display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:7px; font-weight:900; color:#0f2a5f;}
.progress-track{height:9px; border-radius:999px; background:#eaf1fb; overflow:hidden; box-shadow:inset 0 1px 2px rgba(15,42,95,.07);}
.progress-fill{height:100%; border-radius:999px; background:linear-gradient(90deg,#1d4ed8,#38bdf8);}
.status-badge{display:inline-flex; align-items:center; justify-content:center; min-width:86px; border-radius:999px; padding:7px 11px; font-size:12px; font-weight:950; letter-spacing:.04em;}
.status-aman{background:#dcfce7; color:#166534; border:1px solid #86efac;}
.status-waspada{background:#ffedd5; color:#9a3412; border:1px solid #fdba74;}
.status-bahaya{background:#fee2e2; color:#991b1b; border:1px solid #fca5a5;}
.delta-pos{color:#1d4ed8; font-weight:950;}
.delta-neg{color:#dc2626; font-weight:950;}
hr {border-color:#cce0ff;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="hero">
  <div class="hero-title">Dashboard Prediksi TWP90<br>Fintech Lending di Indonesia</div>
  <div class="badge-row">
    <span class="badge">Data terakhir: {last_observed.strftime('%B %Y')}</span>
    <span class="badge">Prediksi mulai: {next_month.strftime('%B %Y')}</span>
    <span class="badge">Model: SARIMAX + XGBoost residual</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Menu telah diubah namanya
MENU_ITEMS = {
    "Dashboard": "Dashboard",
    "Prediksi TWP90": "Prediksi TWP90",
    "Evaluasi Model": "Evaluasi Model",
}
if "selected_menu" not in st.session_state:
    st.session_state["selected_menu"] = "Dashboard"

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-title">Dashboard<br>Prediksi TWP90</div>
            <div class="sidebar-subtitle">Hybrid SARIMAX + XGBoost</div>
            <div class="sidebar-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for menu_key, menu_label in MENU_ITEMS.items():
        label = f"› {menu_label}" if st.session_state["selected_menu"] == menu_key else menu_label
        if st.button(label, key=f"nav_{menu_key}", use_container_width=True):
            st.session_state["selected_menu"] = menu_key
            rerun_dashboard()

selected_menu = st.session_state["selected_menu"]

if selected_menu != "Evaluasi Model":
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    with summary_col1:
        pred_text = "-" if latest_history_pred_pct is None else f"{latest_history_pred_pct:.2f}%"
        pred_note = "Prediksi historis belum tersedia" if latest_history_pred_month is None else f"Periode {latest_history_pred_month.strftime('%B %Y')}"
        st.markdown(value_card("Hasil prediksi terakhir", pred_text, pred_note), unsafe_allow_html=True)
    with summary_col2:
        st.markdown(value_card("TWP90 asli terakhir", f"{last_actual_pct:.2f}%", f"Periode {last_observed.strftime('%B %Y')}", "#0ea5e9"), unsafe_allow_html=True)
    with summary_col3:
        eval_note = eval_summary.get("Source", "Data evaluasi belum tersedia") if eval_summary else "Data evaluasi belum tersedia"
        st.markdown(value_card("Evaluasi MAPE", mape_text, eval_note, "#2563eb"), unsafe_allow_html=True)
    with summary_col4:
        st.markdown(value_card("Data historis", f"{len(raw_history):,} bulan", f"Sampai {last_observed.strftime('%B %Y')}", "#1e3a8a"), unsafe_allow_html=True)


def show_historical_chart():
    fig = go.Figure()
    if not history.empty and "Actual_Original" in history.columns:
        fig.add_trace(go.Scatter(
            x=history["Month"],
            y=history["Actual_Original"] * 100,
            mode="lines+markers",
            name="Aktual historis",
            line=dict(width=3, color="#1e3a8a"),
            marker=dict(size=7, color="#1e3a8a"),
            hovertemplate="%{x|%b %Y}<br>Aktual: %{y:.2f}%<extra></extra>",
        ))
        if "Prediksi_Hybrid_Original" in history.columns:
            fig.add_trace(go.Scatter(
                x=history["Month"],
                y=history["Prediksi_Hybrid_Original"] * 100,
                mode="lines",
                name="Prediksi historis hybrid",
                line=dict(width=2, color="#93c5fd", dash="dot"),
                hovertemplate="%{x|%b %Y}<br>Prediksi: %{y:.2f}%<extra></extra>",
            ))
    else:
        hist = raw_history.reset_index()
        fig.add_trace(go.Scatter(
            x=hist["Month"],
            y=hist[target_col].apply(to_percent_display),
            mode="lines+markers",
            name="Aktual historis",
            line=dict(width=3, color="#1e3a8a"),
            marker=dict(size=7, color="#1e3a8a"),
            hovertemplate="%{x|%b %Y}<br>Aktual: %{y:.2f}%<extra></extra>",
        ))
    fig.add_hline(y=RISK_ORANGE_PCT, line_dash="dash", line_color="#f97316", annotation_text=f"Waspada {RISK_ORANGE_PCT:.0f}%")
    fig.add_hline(y=RISK_RED_PCT, line_dash="dash", line_color="#dc2626", annotation_text=f"Bahaya {RISK_RED_PCT:.0f}%")
    fig.update_layout(
        height=520,
        title=dict(text=""),
        plot_bgcolor="rgba(240,247,255,1)",
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#0f2a5f"),
        xaxis_title="Periode",
        yaxis_title="TWP90 (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=70, b=20),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def show_eval_panel(summary=None, detail=None, source_mode="artifact_history"):
    summary = summary if summary is not None else eval_summary
    detail = detail if detail is not None else eval_detail

    if not summary or detail is None or detail.empty:
        st.markdown(
            """
            <div class="info-box">
                Data evaluasi belum tersedia. Jika ingin evaluasi otomatis dari input user, buka menu <b>Prediksi TWP90</b>, isi seluruh input, lalu klik <b>Hitung Prediksi</b>. Jika ingin evaluasi historis, letakkan <b>test_predictions_hybrid.csv</b> atau <b>dashboard_twp90_history.csv</b> di folder <b>model_artifacts</b> dengan kolom aktual dan prediksi.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if source_mode == "user_prediction":
        st.markdown(
            """
            <div class="info-box">
                Evaluasi ini dihitung dari <b>hasil prediksi terbaru yang diinput/dihitung user</b> pada menu Prediksi TWP90.
                Baris yang belum memiliki TWP90 aktual tidak dihitung agar MAE, RMSE, MAPE, dan R² tetap valid.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="info-box">
                Evaluasi ini dihitung dari data evaluasi historis/artifact model. Setelah user menghitung prediksi baru pada menu Prediksi TWP90 dan tersedia TWP90 aktual input, panel ini otomatis memakai hasil prediksi tersebut.
            </div>
            """,
            unsafe_allow_html=True,
        )

    current_mape = summary.get("MAPE_%")
    current_mape_text = "-" if pd.isna(current_mape) else f"{current_mape:.2f}%"

    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        st.markdown(value_card("MAE", f"{summary['MAE_pp']:.3f} pp", "Rata-rata selisih absolut"), unsafe_allow_html=True)
    with e2:
        st.markdown(value_card("RMSE", f"{summary['RMSE_pp']:.3f} pp", "Rata-rata kuadrat kesalahan antara nilai prediksi dan aktual"), unsafe_allow_html=True)
    with e3:
        st.markdown(value_card("MAPE", current_mape_text, "Persentase rata-rata kesalahan absolut"), unsafe_allow_html=True)
    with e4:
        r2_val = "-" if pd.isna(summary["R2"]) else f"{summary['R2']:.3f}"
        st.markdown(value_card("R²", r2_val, "Kecocokan aktual-prediksi"), unsafe_allow_html=True)
    with e5:
        st.markdown(value_card("Observasi", f"{summary['N']}", summary["Source"]), unsafe_allow_html=True)

    eval_plot = detail.copy()
    fig_eval = go.Figure()
    fig_eval.add_trace(go.Scatter(
        x=eval_plot["Month"] if "Month" in eval_plot.columns else eval_plot.index,
        y=eval_plot["Actual_%"],
        mode="lines+markers",
        name="Aktual",
        line=dict(width=3, color="#1e3a8a"),
        marker=dict(size=7, color="#1e3a8a"),
    ))
    fig_eval.add_trace(go.Scatter(
        x=eval_plot["Month"] if "Month" in eval_plot.columns else eval_plot.index,
        y=eval_plot["Predicted_%"],
        mode="lines+markers",
        name="Prediksi hybrid",
        line=dict(width=3, color="#38bdf8"),
        marker=dict(size=7, color="#38bdf8"),
    ))
    fig_eval.update_layout(
        height=430,
        title="Evaluasi Aktual vs Prediksi Hybrid",
        plot_bgcolor="rgba(240,247,255,1)",
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#0f2a5f"),
        xaxis_title="Periode",
        yaxis_title="TWP90 (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=70, b=20),
        hovermode="x unified",
    )
    st.plotly_chart(fig_eval, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    eval_table = eval_plot.copy()
    if "Month" in eval_table.columns and pd.api.types.is_datetime64_any_dtype(eval_table["Month"]):
        eval_table["Month"] = eval_table["Month"].dt.strftime("%Y-%m")
    eval_table = eval_table[[c for c in ["Month", "Actual_%", "Predicted_%", "Error_pp", "Abs_Error_pp"] if c in eval_table.columns]].copy()
    eval_table = eval_table.rename(columns={
        "Month": "Periode",
        "Actual_%": "TWP90 Aktual (%)",
        "Predicted_%": "Prediksi Hybrid (%)",
        "Error_pp": "Selisih (pp)",
        "Abs_Error_pp": "Abs Error (pp)",
    })
    if "Prediksi Hybrid (%)" in eval_table.columns:
        eval_table["Status Risiko"] = eval_table["Prediksi Hybrid (%)"].apply(
            lambda x: risk_status_from_percent(x, RISK_ORANGE, RISK_RED)
        )

    st.markdown('<div class="eval-table-title">Tabel Detail Evaluasi Aktual vs Prediksi</div>', unsafe_allow_html=True)
    eval_table = clean_undefined_strings(eval_table)
    render_modern_blue_table(
        eval_table,
        progress_cols=[c for c in ["TWP90 Aktual (%)", "Prediksi Hybrid (%)", "Abs Error (pp)"] if c in eval_table.columns],
        delta_cols=[c for c in ["Selisih (pp)"] if c in eval_table.columns],
        status_cols=["Status Risiko"],
        numeric_cols=[c for c in ["Abs Error (pp)"] if c in eval_table.columns],
        suffix_map={"TWP90 Aktual (%)": "%", "Prediksi Hybrid (%)": "%", "Abs Error (pp)": "pp"},
        height=420,
    )


def show_history_table():
    if not history.empty and "Actual_Original" in history.columns:
        table = history.copy()
        table["Aktual_TWP90_%"] = table["Actual_Original"].apply(to_percent_display)
        if "Prediksi_Hybrid_Original" in table.columns:
            table["Prediksi_Hybrid_%"] = table["Prediksi_Hybrid_Original"].apply(to_percent_display)
            table["Selisih_pp"] = table["Prediksi_Hybrid_%"] - table["Aktual_TWP90_%"]
            table["Status"] = table["Prediksi_Hybrid_%"].apply(lambda x: risk_status_from_percent(x, RISK_ORANGE, RISK_RED))
        else:
            table["Status"] = table["Aktual_TWP90_%"].apply(lambda x: risk_status_from_percent(x, RISK_ORANGE, RISK_RED))
        table["Month"] = table["Month"].dt.strftime("%Y-%m")
        selected_cols = ["Month", "Aktual_TWP90_%"]
        if "Prediksi_Hybrid_%" in table.columns:
            selected_cols.extend(["Prediksi_Hybrid_%", "Selisih_pp"])
        selected_cols.append("Status")
        table = table[selected_cols].rename(columns={
            "Month": "Periode",
            "Aktual_TWP90_%": "TWP90 Aktual (%)",
            "Prediksi_Hybrid_%": "Prediksi Hybrid (%)",
            "Selisih_pp": "Selisih (pp)",
            "Status": "Status Risiko",
        })
        table = clean_undefined_strings(table)
    else:
        table = raw_history.reset_index().copy()
        table["Month"] = table["Month"].dt.strftime("%Y-%m")
        table[target_col] = table[target_col].apply(to_percent_display)
        table["Status"] = table[target_col].apply(lambda x: risk_status_from_percent(x, RISK_ORANGE, RISK_RED))
        table = table.rename(columns={
            "Month": "Periode",
            "TWP90 Aktual (%)": target_col,
            "Status": "Status Risiko",
        })
        table = table.rename(columns={target_col: "TWP90 Aktual (%)"})
        table = clean_undefined_strings(table)

    render_modern_history_table(table)


if selected_menu == "Dashboard":
    st.markdown('<div class="section-title" style="margin-top: 1rem;">Tren Historis TWP90</div>', unsafe_allow_html=True)
    show_historical_chart()
    st.markdown('<div class="section-title">Data Historis</div>', unsafe_allow_html=True)
    # Caption setelah judul data historis telah dihapus sesuai permintaan
    show_history_table()

elif selected_menu == "Prediksi TWP90":
    st.markdown('<div class="section-title" style="margin-top: 1rem;">Prediksi TWP90</div>', unsafe_allow_html=True)
    horizon = DASHBOARD_FORECAST_HORIZON

    default_target_date = st.session_state.get(
        "prediction_target_date_persisted",
        to_date(available_target_months[0]),
    )
    default_target_date = max(
        to_date(available_target_months[0]),
        min(default_target_date, to_date(available_target_months[-1])),
    )

    selected_target_date = st.date_input(
        "Pilih periode TWP90 yang ingin diprediksi",
        value=default_target_date,
        min_value=to_date(available_target_months[0]),
        max_value=to_date(available_target_months[-1]),
        format="DD/MM/YYYY",
        help=(
            "Pilih tanggal pada bulan TWP90 yang ingin dicari. Sistem akan membaca bulan dari tanggal yang dipilih. "
            "Untuk menjaga skema one-step-ahead, dashboard meminta variabel eksternal dan TWP90 aktual bulan-bulan sebelum periode target."
        ),
        key="selected_target_calendar",
    )
    st.session_state["prediction_target_date_persisted"] = selected_target_date
    target_month = normalize_month_end(selected_target_date)
    if target_month < available_target_months[0] or target_month > available_target_months[-1]:
        st.error(
            f"Periode target harus berada pada rentang {available_target_months[0].strftime('%B %Y')} "
            f"sampai {available_target_months[-1].strftime('%B %Y')}."
        )
        st.stop()

    input_month = (target_month.to_period("M") - horizon).to_timestamp("M")
    if input_month < next_month:
        st.error(
            f"Periode target {target_month.strftime('%B %Y')} belum dapat dihitung karena input terakhir minimal "
            f"harus {next_month.strftime('%B %Y')}."
        )
        st.stop()

    required_input_count = input_month.to_period("M").ordinal - next_month.to_period("M").ordinal + 1
    future_months = month_range(next_month, required_input_count)
    input_range_text = (
        input_month.strftime("%B %Y")
        if len(future_months) == 1
        else f"{future_months[0].strftime('%B %Y')} s.d. {future_months[-1].strftime('%B %Y')}"
    )
    prediction_range_text = (
        target_month.strftime("%B %Y")
        if len(future_months) == 1
        else f"{future_months[0].strftime('%B %Y')} s.d. {target_month.strftime('%B %Y')}"
    )

    st.markdown(
        f"""
        <div class="info-box">
            Periode TWP90 yang dicari: <b>{target_month.strftime('%B %Y')}</b>.<br>
            Input yang perlu diisi: <b>{input_range_text}</b> berupa variabel eksternal dan <b>TWP90 aktual</b>.<br>
            Hasil yang ditampilkan: prediksi berurutan <b>{prediction_range_text}</b>, sehingga jika memilih Maret maka prediksi bulan-bulan sebelumnya juga muncul.<br>
            TWP90 input dipakai di balik layar untuk memperbarui state SARIMAX dan residual history XGBoost secara one-step-ahead.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(future_months) > 1:
        st.markdown(
            f"""
            <div class="warn-box">
                Untuk mencari TWP90 <b>{target_month.strftime('%B %Y')}</b>, sistem perlu menghitung prediksi berantai dari
                <b>{future_months[0].strftime('%B %Y')}</b> sampai <b>{target_month.strftime('%B %Y')}</b>.
                Karena itu, data eksternal dan TWP90 aktual harus diisi berurutan sampai <b>{future_months[-1].strftime('%B %Y')}</b>
                agar fitur lag/differencing dan residual model tidak melompati periode.
            </div>
            """,
            unsafe_allow_html=True,
        )

    editor_template = prepare_future_input_template(raw_history, future_months, exog_cols, percent_cols)
    current_signature = f"target_{target_month.strftime('%Y-%m')}_input_until_{input_month.strftime('%Y-%m')}_h{horizon}_n{len(future_months)}"

    cached_prediction_input = pd.DataFrame()
    cached_payload = st.session_state.get("prediction_payload")
    if isinstance(cached_payload, dict) and cached_payload.get("signature") == current_signature:
        cached_prediction_input = cached_payload.get("input_df", pd.DataFrame()).copy()
    elif isinstance(st.session_state.get("prediction_input_cache"), dict):
        cached_cache = st.session_state["prediction_input_cache"]
        if cached_cache.get("signature") == current_signature:
            cached_prediction_input = cached_cache.get("input_df", pd.DataFrame()).copy()

    def cached_input_value(month_value, column_name, fallback=0.0):
        if cached_prediction_input is None or cached_prediction_input.empty:
            return float(fallback)
        month_text = month_value.strftime("%Y-%m")
        matched = cached_prediction_input[cached_prediction_input["Month"].astype(str) == month_text]
        if matched.empty or column_name not in matched.columns:
            return float(fallback)
        value = pd.to_numeric(pd.Series([matched.iloc[0][column_name]]), errors="coerce").iloc[0]
        if pd.isna(value):
            return float(fallback)
        return float(value)

    with st.form("prediction_form", clear_on_submit=False):
        input_rows = []
        for i, month in enumerate(future_months):
            month_key = month.strftime("%Y%m")
            expanded = (len(future_months) == 1) or (month == input_month)
            with st.expander(month.strftime("Input %B %Y"), expanded=expanded):
                st.markdown(
                    f"""
                    <div class="input-panel">
                        <div class="eyebrow">Periode input aktual</div>
                        <div class="result-title">{month.strftime('%B %Y')}</div>
                        <div class="result-sub">Isi TWP90 aktual dan estimasi variabel eksternal bulan ini. Kolom persen memakai angka persen asli, bukan desimal. Nilai 0 dianggap belum diisi.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                row = {"Month": month.strftime("%Y-%m")}
                twp_col_left, twp_col_right = st.columns([1, 1])
                with twp_col_left:
                    row[TWP90_INPUT_COL] = st.number_input(
                        "TWP90 aktual bulan ini (%) *",
                        value=cached_input_value(month, TWP90_INPUT_COL, 0.0),
                        step=0.01,
                        format="%.6f",
                        help="Isi nilai TWP90 aktual periode input dalam angka persen asli. Contoh: 4,32% ditulis 4.32.",
                        key=f"num_{current_signature}_{month_key}_{TWP90_INPUT_COL}",
                    )
                with twp_col_right:
                    st.markdown(
                        """
                        <div class="info-box" style="padding:11px 13px; margin-top:3px;">
                            Nilai ini tidak menjadi output target bulan yang sama, tetapi dipakai untuk update residual dan state model sebelum menghitung bulan berikutnya.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                input_columns = st.columns(2)
                template_row = editor_template.loc[editor_template["Month"] == month.strftime("%Y-%m")]
                for j, col in enumerate(exog_cols):
                    default_value = 0.0
                    if not template_row.empty and col in template_row.columns and pd.notna(template_row.iloc[0][col]):
                        default_value = float(template_row.iloc[0][col])
                    default_value = cached_input_value(month, col, default_value)
                    with input_columns[j % 2]:
                        row[col] = st.number_input(
                            f"{make_display_name(col, percent_cols)} *",
                            value=default_value,
                            step=input_step_for_column(col, percent_cols),
                            format="%.6f",
                            help=COLUMN_HELP.get(col, "Isi sesuai skala historis model."),
                            key=f"num_{current_signature}_{month_key}_{col}",
                        )
                input_rows.append(row)

        submitted = st.form_submit_button("Hitung Prediksi", use_container_width=True, type="primary")

    input_snapshot = pd.DataFrame(input_rows)
    current_value_signature = current_signature + "_" + str(pd.util.hash_pandas_object(input_snapshot, index=True).sum())

    if submitted:
        try:
            input_df = input_snapshot.copy()
            validate_required_prediction_inputs(input_df, exog_cols, percent_cols, target_input_col=TWP90_INPUT_COL)
            future_raw_exog = convert_editor_to_future_raw(input_df, exog_cols, percent_cols)
            future_raw_target = convert_editor_to_future_target(input_df, TWP90_INPUT_COL, target_col)
            result, X_sarimax, X_xgb_base, X_xgb_final_used, feature_rows, raw_all, internal_result = predict_hybrid_from_latest_input(
                raw_history, future_raw_exog, artifacts, cfg, input_raw_target=future_raw_target
            )
            st.session_state["prediction_payload"] = {
                "signature": current_signature,
                "value_signature": current_value_signature,
                "target_date": selected_target_date,
                "input_df": input_df,
                "result": result,
                "internal_result": internal_result,
                "X_sarimax": X_sarimax,
                "X_xgb_base": X_xgb_base,
                "X_xgb_final_used": X_xgb_final_used,
                "feature_rows": feature_rows,
                "raw_all": raw_all,
            }
            st.session_state["prediction_input_cache"] = {
                "signature": current_signature,
                "value_signature": current_value_signature,
                "target_date": selected_target_date,
                "input_df": input_df.copy(),
            }
            st.session_state["prediction_target_date_persisted"] = selected_target_date
            st.success("Prediksi berhasil dihitung. Hasil dapat dilihat pada panel di bawah.")
        except Exception as e:
            st.session_state.pop("prediction_payload", None)
            error_message = str(e)
            if "Hasil prediksi tidak ditemukan" not in error_message:
                error_message = f"Prediksi gagal dihitung: {e}"
            st.error(error_message)

    payload = st.session_state.get("prediction_payload")

    if not payload or payload.get("signature") != current_signature or payload.get("value_signature") != current_value_signature:
        st.markdown(
            """
            <div class="panel">
                <div class="section-title">Hasil Prediksi</div>
                <div class="section-help">Hasil prediksi tidak ditemukan untuk pengaturan saat ini. Isi seluruh input wajib dengan nilai selain 0, lalu tekan tombol <b>Hitung Prediksi</b>.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        result = payload["result"]
        input_df = payload["input_df"]

        latest_pred = result.iloc[-1]

        st.markdown('<div class="section-title">Ringkasan Hasil Prediksi</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(value_card("Prediksi periode target", f"{latest_pred['Prediksi_TWP90_%']:.2f}%", f"Periode {latest_pred['Month'].strftime('%B %Y')}", latest_pred["Warna"]), unsafe_allow_html=True)
        with c2:
            st.markdown(value_card("Periode TWP90 dicari", target_month.strftime("%B %Y"), "Output utama dashboard", "#2563eb"), unsafe_allow_html=True)
        with c3:
            st.markdown(value_card("Input aktual sampai", input_month.strftime("%B %Y"), f"{len(future_months)} bulan input", "#0ea5e9"), unsafe_allow_html=True)
        with c4:
            st.markdown(value_card("Status periode target", latest_pred["Status"], latest_pred["Keterangan"], latest_pred["Warna"]), unsafe_allow_html=True)

        selected_row = latest_pred
        selected_status_color = selected_row["Warna"]
        selected_month_text = selected_row["Month"].strftime("%B %Y")

        detail_col1, detail_col2 = st.columns([1.25, 1])
        with detail_col1:
            st.markdown(
                f"""
                <div class="result-panel" style="border-left-color:{selected_status_color};">
                    <div class="eyebrow">Detail periode target</div>
                    <div class="result-title">{selected_month_text}</div>
                    <div class="result-main">{selected_row['Prediksi_TWP90_%']:.2f}%</div>
                    <span class="status-pill" style="background:{selected_status_color};">{selected_row['Status']}</span>
                    <div class="result-sub" style="margin-top:12px;">{selected_row['Keterangan']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with detail_col2:
            st.markdown(
                f"""
                <div class="panel">
                    <div class="eyebrow">Komponen model target</div>
                    <div class="component-grid">
                        <div class="component-item">
                            <div class="component-label">SARIMAX</div>
                            <div class="component-value">{selected_row['Prediksi_SARIMAX_Original'] * 100:.3f}%</div>
                        </div>
                        <div class="component-item">
                            <div class="component-label">Residual XGBoost log</div>
                            <div class="component-value">{selected_row['Prediksi_Residual_XGB_Log']:.6f}</div>
                        </div>
                        <div class="component-item">
                            <div class="component-label">Hybrid original</div>
                            <div class="component-value">{selected_row['Prediksi_Hybrid_Original']:.6f}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        tab_hasil, tab_grafik = st.tabs(["Hasil prediksi", "Grafik tren"])

        with tab_hasil:
            st.markdown('<div class="modern-table-title">Tabel Hasil Prediksi Berantai</div>', unsafe_allow_html=True)
            display_result = result[[
                "Month",
                "Input_Month",
                "Sumber_TWP90",
                "Aktual_TWP90_Input_%",
                "Prediksi_SARIMAX_Original",
                "Prediksi_Residual_XGB_Log",
                "Prediksi_Hybrid_Original",
                "Prediksi_TWP90_%",
                "Error_Hybrid_pp",
                "Status",
            ]].copy()
            display_result["Month"] = display_result["Month"].dt.strftime("%Y-%m")
            display_result["Input_Month"] = pd.to_datetime(display_result["Input_Month"]).dt.strftime("%Y-%m")
            display_result["Prediksi_SARIMAX_%"] = display_result["Prediksi_SARIMAX_Original"] * 100
            display_result["Prediksi_Hybrid_%"] = display_result["Prediksi_Hybrid_Original"] * 100
            display_result = display_result[[
                "Input_Month",
                "Month",
                "Sumber_TWP90",
                "Aktual_TWP90_Input_%",
                "Prediksi_SARIMAX_%",
                "Prediksi_Residual_XGB_Log",
                "Prediksi_Hybrid_%",
                "Prediksi_TWP90_%",
                "Error_Hybrid_pp",
                "Status",
            ]].rename(columns={
                "Input_Month": "Input Aktual Sampai",
                "Month": "Periode Prediksi",
                "Sumber_TWP90": "Jenis Periode",
                "Aktual_TWP90_Input_%": "TWP90 Aktual Input (%)",
                "Prediksi_SARIMAX_%": "SARIMAX (%)",
                "Prediksi_Residual_XGB_Log": "Residual XGB (log)",
                "Prediksi_Hybrid_%": "Prediksi Hybrid (%)",
                "Prediksi_TWP90_%": "Prediksi TWP90 (%)",
                "Error_Hybrid_pp": "Selisih Prediksi-Aktual (pp)",
                "Status": "Status Risiko",
            })
            display_result = clean_undefined_strings(display_result)
            render_modern_blue_table(
                display_result,
                progress_cols=["SARIMAX (%)", "Prediksi Hybrid (%)", "Prediksi TWP90 (%)"],
                delta_cols=["Selisih Prediksi-Aktual (pp)"],
                status_cols=["Status Risiko"],
                numeric_cols=["TWP90 Aktual Input (%)", "Residual XGB (log)"],
                suffix_map={
                    "TWP90 Aktual Input (%)": "%",
                    "SARIMAX (%)": "%",
                    "Prediksi Hybrid (%)": "%",
                    "Prediksi TWP90 (%)": "%",
                    "Selisih Prediksi-Aktual (pp)": "pp",
                },
                height=460,
            )
            csv_download = display_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download hasil prediksi CSV",
                data=csv_download,
                file_name=f"hasil_prediksi_twp90_target_{target_month.strftime('%Y%m')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with tab_grafik:
            fig = go.Figure()
            if not history.empty and "Actual_Original" in history.columns:
                fig.add_trace(go.Scatter(
                    x=history["Month"],
                    y=history["Actual_Original"] * 100,
                    mode="lines+markers",
                    name="Aktual historis",
                    line=dict(width=3, color="#1e3a8a"),
                    marker=dict(size=7, color="#1e3a8a"),
                    hovertemplate="%{x|%b %Y}<br>Aktual: %{y:.2f}%<extra></extra>",
                ))
                if "Prediksi_Hybrid_Original" in history.columns:
                    fig.add_trace(go.Scatter(
                        x=history["Month"],
                        y=history["Prediksi_Hybrid_Original"] * 100,
                        mode="lines",
                        name="Prediksi historis hybrid",
                        line=dict(width=2, color="#93c5fd", dash="dot"),
                        hovertemplate="%{x|%b %Y}<br>Prediksi: %{y:.2f}%<extra></extra>",
                    ))
            else:
                hist = raw_history.reset_index()
                fig.add_trace(go.Scatter(
                    x=hist["Month"],
                    y=hist[target_col].apply(to_percent_display),
                    mode="lines+markers",
                    name="Aktual historis",
                    line=dict(width=3, color="#1e3a8a"),
                    marker=dict(size=7, color="#1e3a8a"),
                ))

            actual_input_plot = result.dropna(subset=["Aktual_TWP90_Input_%"])
            if not actual_input_plot.empty:
                fig.add_trace(go.Scatter(
                    x=actual_input_plot["Month"],
                    y=actual_input_plot["Aktual_TWP90_Input_%"],
                    mode="markers+lines",
                    name="TWP90 aktual input",
                    line=dict(width=2, color="#0f766e", dash="dash"),
                    marker=dict(size=9, color="#0f766e"),
                    hovertemplate="%{x|%b %Y}<br>Aktual input: %{y:.2f}%<extra></extra>",
                ))

            fig.add_trace(go.Scatter(
                x=result["Month"],
                y=result["Prediksi_TWP90_%"],
                mode="markers+text",
                name="Prediksi berantai",
                text=[f"{v:.2f}%" for v in result["Prediksi_TWP90_%"]],
                textposition="top center",
                marker=dict(size=12, color="#38bdf8", line=dict(width=2, color="#0f2a5f")),
                hovertemplate="%{x|%b %Y}<br>Prediksi: %{y:.2f}%<extra></extra>",
            ))
            fig.add_hline(y=RISK_ORANGE_PCT, line_dash="dash", line_color="#f97316", annotation_text=f"Waspada {RISK_ORANGE_PCT:.0f}%")
            fig.add_hline(y=RISK_RED_PCT, line_dash="dash", line_color="#dc2626", annotation_text=f"Bahaya {RISK_RED_PCT:.0f}%")
            fig.update_layout(
                height=540,
                title="Tren Historis, TWP90 Input, dan Prediksi Berantai",
                plot_bgcolor="rgba(240,247,255,1)",
                paper_bgcolor="rgba(255,255,255,0)",
                font=dict(color="#0f2a5f"),
                xaxis_title="Periode",
                yaxis_title="TWP90 (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=70, b=20),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

            status_order = ["AMAN", "WASPADA", "BAHAYA"]
            status_df = result.groupby("Status", as_index=False).size().rename(columns={"size": "Jumlah Bulan"})
            status_df["Status"] = pd.Categorical(status_df["Status"], categories=status_order, ordered=True)
            status_df = status_df.sort_values("Status")
            status_colors_map = {"AMAN": "#16a34a", "WASPADA": "#f97316", "BAHAYA": "#dc2626"}
            marker_colors = [status_colors_map.get(s, "#2563eb") for s in status_df["Status"].astype(str)]
            fig_status = go.Figure(go.Bar(
                x=status_df["Status"].astype(str),
                y=status_df["Jumlah Bulan"],
                text=status_df["Jumlah Bulan"],
                textposition="outside",
                marker_color=marker_colors,
            ))
            fig_status.update_layout(
                height=330,
                title="Distribusi Status Risiko pada Hasil Prediksi Berantai",
                plot_bgcolor="rgba(240,247,255,1)",
                paper_bgcolor="rgba(255,255,255,0)",
                font=dict(color="#0f2a5f"),
                xaxis_title="Status",
                yaxis_title="Jumlah Bulan",
                margin=dict(l=20, r=20, t=65, b=20),
            )
            st.plotly_chart(fig_status, use_container_width=True, config={"displayModeBar": False, "responsive": True})


elif selected_menu == "Evaluasi Model":
    st.markdown('<div class="section-title" style="margin-top: 1rem;">Evaluasi Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-help">Panel ini menampilkan evaluasi aktual vs prediksi, error, absolute error, dan feature importance model hybrid.</div>', unsafe_allow_html=True)
    active_eval_summary, active_eval_detail, active_eval_scale, active_eval_source = get_active_evaluation_dataset(eval_raw)
    show_eval_panel(active_eval_summary, active_eval_detail, active_eval_source)
    st.markdown('<div class="feature-importance-spacer"></div>', unsafe_allow_html=True)
    show_hybrid_feature_importance()
