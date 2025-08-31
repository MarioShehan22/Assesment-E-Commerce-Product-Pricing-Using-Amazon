from __future__ import annotations

import io
import os
import re
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------
# Config
# -------------------------
MODEL_PATH = os.getenv("PRICE_MODEL_PATH", "price_model.joblib")
CSV_PATH = os.getenv("PRICE_DATA_CSV", "amazon_products.csv")
RANDOM_STATE = 42

# -------------------------
# Helpers
# -------------------------
def parse_price(value) -> float:
    """Parse price/sold_quantity strings to float. Handles commas, ranges, K/M."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).replace(",", "").strip()
    m = re.search(r"([0-9]*\.?[0-9]+)(?:\s*[-–]\s*([0-9]*\.?[0-9]+))?\s*([KkMm])?", s)
    if not m:
        return np.nan
    a, b, suf = m.groups()
    x = (float(a) + float(b)) / 2 if b else float(a)
    if suf:
        if suf.upper() == "K":
            x *= 1_000
        elif suf.upper() == "M":
            x *= 1_000_000
    return float(x)

def load_dataframe_from_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="ISO-8859-1", on_bad_lines="skip")

    # case-insensitive check for required columns
    expected = {"title", "rating", "sold_quantity", "price"}
    lower_map = {c.lower(): c for c in df.columns}
    missing = expected - set(lower_map.keys())
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}. Expected: {sorted(expected)}")

    # normalize names
    df = df.rename(columns={lower_map["title"]: "title",
                            lower_map["rating"]: "rating",
                            lower_map["sold_quantity"]: "sold_quantity",
                            lower_map["price"]: "price"})

    # clean/parse
    df["title"] = df["title"].fillna("Unknown").astype(str).str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["sold_quantity"] = df["sold_quantity"].apply(parse_price)
    df["price"] = df["price"].apply(parse_price)
    df = df.dropna(subset=["title", "rating", "sold_quantity", "price"]).copy()
    df["rating"] = df["rating"].clip(0, 5)
    return df

def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    text_tf = TfidfVectorizer(max_features=10_000, min_df=2, ngram_range=(1, 2))
    num_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median"))] +
        ([("scale", StandardScaler())] if scale_numeric else [])
    )
    pre = ColumnTransformer(
        [("title", text_tf, "title"),
         ("num", num_pipe, ["rating", "sold_quantity"])],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return pre

def build_model() -> Pipeline:
    # RF doesn't need scaling
    pre = build_preprocessor(scale_numeric=False)
    rf = RandomForestRegressor(
        n_estimators=350,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("model", rf)])

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

# -------------------------
# App & State
# -------------------------
app = FastAPI(title="Price Optimizer API", version="1.0.0")

MODEL: Optional[Pipeline] = None
MODEL_INFO: dict = {
    "ready": False,
    "source": None,        # 'joblib' | 'csv' | 'dummy'
    "trained_at": None,
    "n_train": 0,
    "metrics": None,
}

def _set_model(m: Pipeline, source: str, n_train: int = 0, metrics: Optional[dict] = None):
    global MODEL, MODEL_INFO
    MODEL = m
    MODEL_INFO = {
        "ready": True,
        "source": source,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_train": int(n_train),
        "metrics": metrics,
    }

def bootstrap_model():
    # 1) Try load pre-trained
    if os.path.exists(MODEL_PATH):
        try:
            m = joblib.load(MODEL_PATH)
            _set_model(m, source="joblib", n_train=0, metrics=None)
            return
        except Exception as e:
            print(f"Failed to load {MODEL_PATH}: {e}")
    # 2) Train from CSV if present
    try:
        df = load_dataframe_from_csv(CSV_PATH)
        X = df[["title", "rating", "sold_quantity"]].copy()
        y = df["price"].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        pipe = build_model()
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        mets = evaluate(y_te, preds)
        joblib.dump(pipe, MODEL_PATH)
        _set_model(pipe, source="csv", n_train=len(X_tr), metrics=mets)
        return
    except Exception as e:
        print(f"Could not train from CSV: {e}")
    # 3) Fallback dummy
    dummy_pre = build_preprocessor(scale_numeric=False)
    dummy = DummyRegressor(strategy="constant", constant=0.0)
    pipe = Pipeline([("pre", dummy_pre), ("model", dummy)])
    df_min = pd.DataFrame([{"title": "Unknown", "rating": 0.0, "sold_quantity": 0.0, "price": 0.0}])
    pipe.fit(df_min[["title", "rating", "sold_quantity"]], df_min["price"])
    _set_model(pipe, source="dummy", n_train=1, metrics=None)

@app.on_event("startup")
def _on_startup():
    bootstrap_model()

# -------------------------
# API Schemas
# -------------------------
class PredictRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    rating: float = Field(...)
    sold_quantity: float = Field(...)

class PredictResponse(BaseModel):
    price: float
    model_source: str

# -------------------------
# Routes
# -------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=_HTML_PAGE, status_code=200)

@app.get("/status")
def status():
    return JSONResponse(MODEL_INFO)

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    # simple validation (no Pydantic v2 dependency)
    title = (payload.title or "").strip()
    if not title:
        raise HTTPException(status_code=422, detail="title cannot be empty")
    try:
        rating = float(payload.rating)
        sold = float(payload.sold_quantity)
    except Exception:
        raise HTTPException(status_code=422, detail="rating and sold_quantity must be numbers")
    if not (0.0 <= rating <= 5.0):
        raise HTTPException(status_code=422, detail="rating must be between 0 and 5")
    if sold < 0:
        raise HTTPException(status_code=422, detail="sold_quantity must be non-negative")

    row = pd.DataFrame([{"title": title, "rating": rating, "sold_quantity": sold}])
    try:
        pred = float(np.round(MODEL.predict(row)[0], 2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return PredictResponse(price=pred, model_source=MODEL_INFO.get("source") or "dummy")

@app.post("/retrain")
def retrain(file: UploadFile = File(...)):
    """Upload a CSV to retrain the model. Expected columns: title, rating, sold_quantity, price"""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")
    try:
        content = file.file.read()
        df = pd.read_csv(io.BytesIO(content))

        # case-insensitive columns
        df.columns = [c.lower() for c in df.columns]
        required = {"title", "rating", "sold_quantity", "price"}
        if not required.issubset(set(df.columns)):
            raise ValueError("CSV must include columns: title, rating, sold_quantity, price")

        # normalize & parse
        df["sold_quantity"] = df["sold_quantity"].apply(parse_price)
        df["price"] = df["price"].apply(parse_price)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").clip(0, 5)
        df["title"] = df["title"].fillna("Unknown").astype(str).str.strip()
        df = df.dropna(subset=["title", "rating", "sold_quantity", "price"]).copy()
        if len(df) < 20:
            raise ValueError("Need at least 20 valid rows to train.")

        X = df[["title", "rating", "sold_quantity"]]
        y = df["price"].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        pipe = build_model()
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        mets = evaluate(y_te, preds)
        joblib.dump(pipe, MODEL_PATH)
        _set_model(pipe, source="csv", n_train=len(X_tr), metrics=mets)
        return {"ok": True, "trained": len(X_tr), "metrics": mets}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Retrain failed: {e}")
    finally:
        try:
            file.file.close()
        except Exception:
            pass

# -------------------------
# HTML UI (no @apply, works with CDN)
# -------------------------
_HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Price Optimizer</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-50 via-indigo-50 to-white text-slate-800">
  <div class="max-w-3xl mx-auto p-6">
    <header class="flex items-center justify-between mb-6">
      <div>
        <h1 class="text-2xl md:text-3xl font-semibold">Price Optimizer</h1>
        <p class="text-slate-600">Estimate a product's price from its title, rating, and sold quantity.</p>
      </div>
      <div id="status" class="inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm bg-slate-100 text-slate-700">Loading model…</div>
    </header>

    <div class="grid gap-6">
      <section class="bg-white/90 backdrop-blur rounded-2xl shadow p-6">
        <div class="grid gap-4">
          <label class="block">
            <span class="block text-sm font-medium text-slate-700">Product title</span>
            <input id="title" class="w-full rounded-xl border border-gray-300 px-4 py-2 focus:ring-2 focus:ring-indigo-500 focus:outline-none" type="text" placeholder="e.g. Wireless Noise-Cancelling Headphones with Mic" />
          </label>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <label class="block">
              <span class="block text-sm font-medium text-slate-700">Rating (0–5)</span>
              <input id="rating" class="w-full rounded-xl border border-gray-300 px-4 py-2 focus:ring-2 focus:ring-indigo-500 focus:outline-none" type="number" min="0" max="5" step="0.1" placeholder="4.6" />
            </label>
            <label class="block">
              <span class="block text-sm font-medium text-slate-700">Sold quantity</span>
              <input id="sold" class="w-full rounded-xl border border-gray-300 px-4 py-2 focus:ring-2 focus:ring-indigo-500 focus:outline-none" type="number" min="0" step="1" placeholder="1250" />
            </label>
          </div>

          <div class="flex items-center gap-3">
            <button id="predictBtn" class="rounded-xl px-4 py-2 font-medium shadow hover:shadow-md transition active:scale-95 bg-indigo-600 text-white">Predict price</button>
            <span id="predicting" class="hidden text-slate-500">Predicting…</span>
          </div>

          <div id="result" class="mt-2 text-xl font-semibold"></div>
          <div id="error" class="mt-2 text-sm text-rose-600"></div>
        </div>
      </section>

      <details class="bg-white/90 backdrop-blur rounded-2xl shadow p-6">
        <summary class="cursor-pointer font-medium">Train / Retrain with CSV</summary>
        <div class="mt-4 grid gap-3">
          <p class="text-sm text-slate-600">Upload a CSV with columns: <code>title</code>, <code>rating</code>, <code>sold_quantity</code>, <code>price</code>.</p>
          <input id="csv" type="file" accept=".csv" class="w-full rounded-xl border border-gray-300 px-4 py-2 focus:ring-2 focus:ring-indigo-500 focus:outline-none" />
          <div class="flex items-center gap-3">
            <button id="retrainBtn" class="rounded-xl px-4 py-2 font-medium shadow hover:shadow-md transition active:scale-95 bg-slate-800 text-white">Upload & retrain</button>
            <span id="retraining" class="hidden text-slate-500">Training… this may take a bit.</span>
          </div>
          <pre id="metrics" class="mt-2 bg-slate-50 rounded-xl p-3 text-sm overflow-auto"></pre>
        </div>
      </details>

      <footer class="text-xs text-slate-500 text-center">Built with FastAPI &amp; scikit-learn</footer>
    </div>
  </div>

<script>
async function fetchStatus(){
  try{
    const res = await fetch('/status');
    const s = await res.json();
    const badge = document.getElementById('status');
    const src = s.source || 'unknown';
    const emoji = src==='joblib' ? '💾' : (src==='csv' ? '🧠' : '🪄');
    const r2 = s.metrics && (typeof s.metrics.R2 === 'number') ? s.metrics.R2.toFixed(3) : '';
    badge.textContent = emoji + ' Model: ' + src + (r2 ? ' • R² ' + r2 : '');
    badge.className = 'inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm ' + (s.ready ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700');
  }catch(e){ console.error(e); }
}

function showError(msg){ document.getElementById('error').textContent = msg || ''; }
function showResult(val){
  const el = document.getElementById('result');
  el.textContent = (val === null || val === undefined) ? '' : ('Predicted price: ' + val);
}

async function onPredict(){
  showError(''); showResult(null);
  document.getElementById('predicting').classList.remove('hidden');
  try{
    const title = document.getElementById('title').value.trim();
    const rating = parseFloat(document.getElementById('rating').value);
    const sold = parseFloat(document.getElementById('sold').value);
    if(!title){ throw new Error('Please enter a product title'); }
    if(isNaN(rating) || rating<0 || rating>5){ throw new Error('Rating must be between 0 and 5'); }
    if(isNaN(sold) || sold<0){ throw new Error('Sold quantity must be a non-negative number'); }
    const res = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({title, rating, sold_quantity: sold})});
    if(!res.ok){ const err = await res.json(); throw new Error(err.detail || 'Prediction failed'); }
    const data = await res.json(); showResult(data.price);
  }catch(e){ showError(e.message || String(e)); }
  finally{ document.getElementById('predicting').classList.add('hidden'); }
}

async function onRetrain(){
  showError('');
  document.getElementById('retraining').classList.remove('hidden');
  try{
    const file = document.getElementById('csv').files[0];
    if(!file){ throw new Error('Choose a CSV first'); }
    const fd = new FormData(); fd.append('file', file);
    const res = await fetch('/retrain', { method:'POST', body: fd });
    const text = await res.text();
    if(!res.ok){ throw new Error(JSON.parse(text).detail || 'Retraining failed'); }
    const data = JSON.parse(text);
    document.getElementById('metrics').textContent = JSON.stringify(data, null, 2);
    await fetchStatus();
  }catch(e){ showError(e.message || String(e)); }
  finally{ document.getElementById('retraining').classList.add('hidden'); }
}

document.getElementById('predictBtn').addEventListener('click', onPredict);
document.getElementById('retrainBtn').addEventListener('click', onRetrain);
fetchStatus();
</script>
</body>
</html>"""