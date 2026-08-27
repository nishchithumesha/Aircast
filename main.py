import io, os, time, math, itertools, tempfile, random
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import requests
import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st

from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # kept only for potential future use

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# NEW: Google Gen AI SDK (replaces google.generativeai)
from google import genai
from google.genai import types as genai_types


# -------------------------------
# Chatbot API configuration
# -------------------------------
CHATBOT_API_KEY = "AIzaSyBiYNDMticjUhBzOUO41-J9HHJB5nUJQH8" 
# client for Gemini Developer API
chat_client = genai.Client(api_key=CHATBOT_API_KEY)

CHATBOT_MODEL_ID = "gemini-2.5-flash"   # modern replacement for 1.5 flash


# -------------------------------
# Open-Meteo Air Quality helpers
# -------------------------------
OM_BASE = "https://air-quality-api.open-meteo.com/v1/air-quality"
DEFAULT_VARS = ["pm2_5","pm10","nitrogen_dioxide","ozone","sulphur_dioxide","carbon_monoxide"]


def classify_aqi(aqi: float):
    """
    Color ranges we use for ANY selected pollutant value:
    - 0–50   : Good (green)       → good environment
    - 51–100 : Moderate (orange)
    - 101–200: Bad (orange)
    - >200   : Critical (red)
    """
    if aqi <= 50:
        return "Good", "green", "Air quality is good (0–50). This is considered a good environment."
    elif aqi <= 100:
        return "Moderate", "orange", "Air quality is acceptable (51–100), but sensitive groups should be cautious."
    elif aqi <= 200:
        return "Bad", "orange", "Air quality is poor (101–200). Limit prolonged outdoor activities."
    else:
        return "Critical", "red", "Air quality is very unhealthy/critical (>200). Avoid outdoor activities if possible."


def fetch_hourly(lat: float, lon: float, past_days: int = 3,
                 variables: Optional[List[str]] = None,
                 timezone: str = "auto",
                 timeout: float = 20.0) -> Dict[str, Any]:
    vars_list = variables or DEFAULT_VARS
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(vars_list),
        "past_days": max(1, min(7, int(past_days))),
        "timezone": timezone
    }
    r = requests.get(OM_BASE, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def om_to_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame(columns=["time","variable","value","unit"])
    rows = []
    units = payload.get("hourly_units", {})
    for var, series in hourly.items():
        if var == "time":
            continue
        for t, v in zip(times, series):
            rows.append({"time": t, "variable": var, "value": v, "unit": units.get(var, "")})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
    return df.sort_values("time")


# -------------------------------
# Satellite tile helpers (EOX, GIBS, ESRI)
# -------------------------------
def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    )
    return xtile, ytile


def num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float,float]:
    n = 2.0**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


EOX_BASE = "https://tiles.maps.eox.at"
GIBS_BASE = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best"
ESRI_BASE = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile"


def fetch_tile_eox(z: int, x: int, y: int, timeout: float = 10.0) -> Optional[Image.Image]:
    layers = [
        "s2cloudless-2020_3857",
        "s2cloudless_3857",
    ]
    for layer in layers:
        url = f"{EOX_BASE}/{layer}/{z}/{y}/{x}.jpg"
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            continue
    return None


def fetch_tile_gibs(z: int, x: int, y: int, timeout: float = 10.0) -> Optional[Image.Image]:
    layers = [
        "MODIS_Terra_CorrectedReflectance_TrueColor/default/GoogleMapsCompatible_Level9",
        "VIIRS_SNPP_CorrectedReflectance_TrueColor/default/GoogleMapsCompatible_Level9"
    ]
    for layer in layers:
        url = f"{GIBS_BASE}/{layer}/{z}/{y}/{x}.jpg"
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            continue
    return None


def fetch_tile_esri(z: int, x: int, y: int, timeout: float = 10.0) -> Optional[Image.Image]:
    url = f"{ESRI_BASE}/{z}/{y}/{x}"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None
    return None


def fetch_satellite_tile(z: int, x: int, y: int, timeout: float = 10.0) -> Tuple[Optional[Image.Image], str]:
    img = fetch_tile_eox(z, x, y, timeout=timeout)
    if img is not None:
        return img, "EOX"
    img = fetch_tile_gibs(z, x, y, timeout=timeout)
    if img is not None:
        return img, "GIBS"
    img = fetch_tile_esri(z, x, y, timeout=timeout)
    if img is not None:
        return img, "ESRI"
    return None, "NONE"


def stitched_satellite(center_lon: float, center_lat: float,
                       z: int = 12,
                       grid: int = 3,
                       tile_px: int = 256,
                       timeout: float = 10.0) -> Tuple[Optional[Image.Image], str, bool]:
    x_center, y_center = deg2num(center_lat, center_lon, z)
    half = grid // 2
    provider_used = None
    canvas_img = Image.new("RGB", (grid * tile_px, grid * tile_px), (0, 0, 0))
    any_ok = False

    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            xt = x_center + dx
            yt = y_center + dy
            tile, prov = fetch_satellite_tile(z, xt, yt, timeout=timeout)
            if tile is not None:
                if provider_used is None:
                    provider_used = prov
                tile = tile.resize((tile_px, tile_px))
                cx = (dx + half) * tile_px
                cy = (dy + half) * tile_px
                canvas_img.paste(tile, (cx, cy))
                any_ok = True

    if not any_ok:
        return None, "NONE", False
    if provider_used is None:
        provider_used = "MIXED"
    return canvas_img, provider_used, True


def test_one_tile(lon: float, lat: float, z: int = 12) -> Dict[str, Any]:
    x, y = deg2num(lat, lon, z)
    info = {}
    for name, func in [
        ("EOX", fetch_tile_eox),
        ("GIBS", fetch_tile_gibs),
        ("ESRI", fetch_tile_esri),
    ]:
        t0 = time.time()
        img = func(z, x, y)
        dt = time.time() - t0
        info[name] = {"ok": img is not None, "dt": dt}
    return info


# -------------------------------
# Summary + Forecast + PDF + Chatbot helper
# -------------------------------
def ask_chatbot(question: str, context: str = "", history_text: str = "") -> str:
    """
    Use the AI chatbot to answer questions about air quality.
    Uses google-genai with gemini-2.5-flash.
    """
    try:
        prompt = (
            "You are a friendly chatbot that explains air quality data and health impact "
            "in very simple language. Avoid technical jargon and give practical tips.\n\n"
            "Here is the conversation so far:\n"
            f"{history_text}\n\n"
            "Use the context (CSV air-quality data) only for numbers and trends.\n\n"
            f"Context (CSV):\n{context}\n\n"
            f"User question:\n{question}"
        )

        response = chat_client.models.generate_content(
            model=CHATBOT_MODEL_ID,
            contents=prompt
        )
        # google-genai uses .text for text output
        return response.text
    except Exception as e:
        return f"Error from assistant: {e}"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["variable","count","mean","median","min","max","std"])
    g = df.groupby("variable")["value"]
    return pd.DataFrame({
        "count": g.count(), "mean": g.mean(), "median": g.median(),
        "min": g.min(), "max": g.max(), "std": g.std()
    }).reset_index()


def simple_forecast(df: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
    """
    Very basic linear regression forecast per variable on time index.
    Safely drops NaN values so scikit-learn does not crash.
    """
    if df.empty:
        return pd.DataFrame(columns=["time","variable","value"])
    out_rows = []
    for var in df["variable"].unique():
        dff = df[df["variable"] == var].sort_values("time").copy()
        dff = dff.dropna(subset=["time", "value"])
        if len(dff) < 4:
            continue
        t0 = dff["time"].min()
        x = (dff["time"] - t0).dt.total_seconds().values.reshape(-1, 1)
        y = dff["value"].values.reshape(-1, 1)
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        model = LinearRegression()
        model.fit(x, y)
        last_time = dff["time"].max()
        for h in range(1, horizon_hours + 1):
            t_future = last_time + pd.Timedelta(hours=h)
            x_future = np.array([(t_future - t0).total_seconds()]).reshape(-1, 1)
            y_future = model.predict(x_future)[0, 0]
            out_rows.append({"time": t_future, "variable": var, "value": float(y_future)})
    out = pd.DataFrame(out_rows)
    return out


def build_pdf(df_raw: pd.DataFrame, df_summary: pd.DataFrame, title: str,
              meta: Dict[str, Any],
              sat_png_path: Optional[str] = None,
              sat_note: str = "",
              forecast_summary: Optional[pd.DataFrame] = None,
              forecast_wide: Optional[pd.DataFrame] = None,
              sample_rows: Optional[pd.DataFrame] = None) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # Cover
    c.setFont("Helvetica-Bold", 20)
    c.drawString(2 * cm, height - 2 * cm, title)
    c.setFont("Helvetica", 11)
    y = height - 3 * cm
    for k, v in meta.items():
        c.drawString(2 * cm, y, f"{k}: {v}")
        y -= 0.6 * cm
    c.showPage()

    # Satellite
    if sat_png_path and os.path.exists(sat_png_path):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(2 * cm, height - 2 * cm, "Satellite mosaic (raw)")
        c.setFont("Helvetica", 10)
        c.drawString(2 * cm, height - 2.5 * cm, sat_note or "")
        img = ImageReader(sat_png_path)
        iw, ih = img.getSize()
        max_w, max_h = width - 4 * cm, height - 5 * cm
        scale = min(max_w / iw, max_h / ih)
        dw, dh = iw * scale, ih * scale
        c.drawImage(img, (width - dw)/2, (height - dh)/2 - 1 * cm,
                    width=dw, height=dh, preserveAspectRatio=True)
        c.showPage()

    # Summary table
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, height - 2 * cm, "Summary statistics")
    c.setFont("Helvetica", 9)
    y = height - 3 * cm
    if df_summary is not None and not df_summary.empty:
        cols = list(df_summary.columns)
        col_width = (width - 4 * cm) / len(cols)
        for i, col in enumerate(cols):
            c.drawString(2 * cm + i * col_width, y, col)
        y -= 0.5 * cm
        for _, row in df_summary.iterrows():
            for i, col in enumerate(cols):
                val = str(row[col])[:12]
                c.drawString(2 * cm + i * col_width, y, val)
            y -= 0.5 * cm
            if y < 2 * cm:
                c.showPage()
                c.setFont("Helvetica", 9)
                y = height - 2 * cm
    else:
        c.drawString(2 * cm, y, "No summary available.")
    c.showPage()

    # Forecast summary
    if forecast_summary is not None and not forecast_summary.empty:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(2 * cm, height - 2 * cm, "Forecast (summary)")
        c.setFont("Helvetica", 9)
        y = height - 3 * cm
        cols = list(forecast_summary.columns)
        col_width = (width - 4 * cm) / len(cols)
        for i, col in enumerate(cols):
            c.drawString(2 * cm + i * col_width, y, col)
        y -= 0.5 * cm
        for _, row in forecast_summary.iterrows():
            for i, col in enumerate(cols):
                val = str(row[col])[:12]
                c.drawString(2 * cm + i * col_width, y, val)
            y -= 0.5 * cm
            if y < 2 * cm:
                c.showPage()
                c.setFont("Helvetica", 9)
                y = height - 2 * cm
        c.showPage()

    # Forecast wide
    if forecast_wide is not None and not forecast_wide.empty:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(2 * cm, height - 2 * cm, "Forecast (wide table, head)")
        c.setFont("Helvetica", 8)
        y = height - 3 * cm
        dfw = forecast_wide.head(20)
        cols = list(dfw.columns)
        col_width = (width - 4 * cm) / len(cols)
        for i, col in enumerate(cols):
            c.drawString(2 * cm + i * col_width, y, col[:10])
        y -= 0.5 * cm
        for _, row in dfw.iterrows():
            for i, col in enumerate(cols):
                val = str(row[col])[:10]
                c.drawString(2 * cm + i * col_width, y, val)
            y -= 0.45 * cm
            if y < 2 * cm:
                c.showPage()
                c.setFont("Helvetica", 8)
                y = height - 2 * cm
        c.showPage()

    # Sample raw rows
    if sample_rows is not None and not sample_rows.empty:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(2 * cm, height - 2 * cm, "Sample raw rows")
        c.setFont("Helvetica", 8)
        y = height - 3 * cm
        dfr = sample_rows.head(40)
        cols = list(dfr.columns)
        col_width = (width - 4 * cm) / len(cols)
        for i, col in enumerate(cols):
            c.drawString(2 * cm + i * col_width, y, col[:10])
        y -= 0.5 * cm
        for _, row in dfr.iterrows():
            for i, col in enumerate(cols):
                val = str(row[col])[:10]
                c.drawString(2 * cm + i * col_width, y, val)
            y -= 0.45 * cm
            if y < 2 * cm:
                c.showPage()
                c.setFont("Helvetica", 8)
                y = height - 2 * cm
        c.showPage()

    c.save()
    buf.seek(0)
    return buf.read()


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AeroVision — AQ + Satellite (RAW) + Forecast tables", layout="wide")
st.title("AeroVision — Online AQ + Satellite (RAW) + Forecast Tables")

# Initialize chat history (for WhatsApp-style messages)
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list of {"role": "user"/"assistant", "text": str}

# Small always-visible legend for your color logic
with st.expander("Status color legend (for project explanation)", expanded=True):
    st.success("🟢 Good: 0–50  — good environment")
    st.warning("🟠 Moderate / Bad: 51–200  — air not ideal, some health concerns")
    st.error("🔴 Critical: >200  — very unhealthy / critical condition")


# Sidebar: controls + chatbot (separate)
with st.sidebar:
    st.header("Controls")
    c1, c2 = st.columns(2)
    lat = c1.text_input("Latitude", "12.9716")
    lon = c2.text_input("Longitude", "77.5946")
    past_days = st.slider("Past days (1–7)", 1, 7, 3)
    vars_sel = st.multiselect(
        "Variables", DEFAULT_VARS,
        default=["pm2_5","pm10","nitrogen_dioxide","ozone","sulphur_dioxide"]
    )

    st.divider()
    st.caption("Satellite")
    sat_zoom = st.slider("Zoom (6–15)", 6, 15, 12)
    sat_grid = st.slider("Grid size (tiles/side)", 1, 5, 3)
    sat_tile_px = st.radio("Tile size (px)", [256, 512], index=0)

    st.divider()
    st.subheader("🤖 Chatbot assistant")

    chat_question = st.text_area(
        "Type your question:", key="chatbot_question", height=80
    )
    chat_ask = st.button("Send", key="chatbot_ask")

    st.markdown("---")
    st.markdown("*Chat history*")
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f"🧑 *You:* {msg['text']}")
        else:
            st.markdown(f"🤖 *Assistant:* {msg['text']}")


colA, colB = st.columns(2)
if colA.button("Test satellite connectivity"):
    try:
        lat_f_test, lon_f_test = float(lat), float(lon)
        res = test_one_tile(lon_f_test, lat_f_test, z=int(sat_zoom))
        st.write(res)
    except Exception as e:
        st.error(f"Satellite test failed: {e}")

run_clicked = colB.button("Run", type="primary")

# Always parse lat/lon so we can use them even when just chatting
try:
    lat_f, lon_f = float(lat), float(lon)
except ValueError:
    st.error("Latitude/Longitude must be numeric.")
    st.stop()

# When "Run" is clicked, fetch fresh data and store in session_state
if run_clicked:
    with st.spinner("Calling Open-Meteo (hourly AQ)…"):
        try:
            payload = fetch_hourly(
                lat_f, lon_f,
                past_days=past_days,
                variables=vars_sel or DEFAULT_VARS
            )
            df = om_to_dataframe(payload)
        except Exception as e:
            st.error(f"AQ fetch failed: {e}")
            df = pd.DataFrame()
    if df.empty:
        st.warning("No AQ data returned. Try different coordinates or fewer days.")
    else:
        st.success(f"AQ data OK — {len(df)} hourly points across {df['variable'].nunique()} pollutants.")
        st.session_state["df"] = df
        st.session_state["meta"] = {
            "Latitude": lat_f,
            "Longitude": lon_f,
            "Past days": past_days,
            "Variables": ", ".join(vars_sel or DEFAULT_VARS),
            "Satellite zoom": sat_zoom,
            "Satellite grid": sat_grid,
            "Generated (UTC)": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        }

# Chatbot logic (WhatsApp style, stays in sidebar, but uses data if available)
df = st.session_state.get("df", None)
if chat_ask and chat_question.strip():
    if df is not None and not df.empty:
        recent_df = df.sort_values("time").tail(48)
        context_csv = recent_df.to_csv(index=False)
    else:
        context_csv = ""

    history_text = ""
    for msg in st.session_state["chat_history"]:
        prefix = "User:" if msg["role"] == "user" else "Assistant:"
        history_text += f"{prefix} {msg['text']}\n"

    with st.spinner("Assistant is typing..."):
        answer = ask_chatbot(chat_question, context=context_csv, history_text=history_text)

    st.session_state["chat_history"].append({"role": "user", "text": chat_question})
    st.session_state["chat_history"].append({"role": "assistant", "text": answer})


# If we already have data in session_state, show full UI
df = st.session_state.get("df", None)

if df is not None and not df.empty:
    # ----- Colored warning based on SELECTED VARIABLE -----
    available_vars = list(df["variable"].unique())
    selected_var_for_status = None

    for v in vars_sel:
        if v in available_vars:
            selected_var_for_status = v
            break
    if selected_var_for_status is None:
        if "pm2_5" in available_vars:
            selected_var_for_status = "pm2_5"
        elif available_vars:
            selected_var_for_status = available_vars[0]

    if selected_var_for_status is not None:
        d_status = df[df["variable"] == selected_var_for_status].sort_values("time")
        vals = d_status["value"].dropna()
        if not vals.empty:
            latest_val = float(vals.iloc[-1])
            unitv = d_status["unit"].mode().iat[0] if not d_status["unit"].empty else ""
            status, color, msg = classify_aqi(latest_val)

            if color == "green":
                box = st.success
            elif color == "orange":
                box = st.warning
            else:
                box = st.error

            box(
                f"**Status for {selected_var_for_status}: {status}**  |  "
                f"Latest value: *{latest_val:.1f} {unitv}*\n\n{msg}"
            )
            st.caption(
                "We apply the same ranges (0–50, 51–100, 101–200, >200) "
                "to the selected variable's latest value to show Good / Moderate / Bad / Critical."
            )
        else:
            st.info(f"No valid (non-NaN) values found for {selected_var_for_status}.")
    else:
        st.info("No suitable variable found for status indicator.")

    # Satellite mosaic
    with st.spinner("Fetching satellite mosaic…"):
        sat_img, provider_used, ok = stitched_satellite(
            lon_f, lat_f, z=int(sat_zoom), grid=int(sat_grid), tile_px=int(sat_tile_px)
        )
        sat_png = None
        if not ok or sat_img is None:
            st.warning("Satellite mosaic failed. Check connectivity or try another zoom.")
        else:
            st.info(f"Satellite provider used: {provider_used}")
            st.image(
                sat_img,
                caption=f"Raw satellite mosaic (provider: {provider_used})",
                use_container_width=True
            )
            tmp_dir = tempfile.gettempdir()
            sat_png = os.path.join(tmp_dir, f"aircast_sat_{int(time.time())}.png")
            sat_img.save(sat_png)

    # Hourly time-series (historical)
    st.subheader("Hourly time-series (historical)")
    for v in sorted(df["variable"].unique()):
        dff = df[df["variable"] == v]
        unitv = dff["unit"].mode().iat[0] if not dff["unit"].empty else ""
        fig = px.line(dff, x="time", y="value", title=f"{v} ({unitv}) — {len(dff)} points")
        st.plotly_chart(fig, use_container_width=True)

    # Summary stats table
    st.subheader("Summary statistics")
    summ = summarize(df)
    st.dataframe(summ.sort_values("variable"), use_container_width=True)

    # CSV downloads
    cA, cB = st.columns(2)
    with cA:
        st.download_button(
            "Download CSV (raw hourly)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="openmeteo_raw.csv",
            mime="text/csv"
        )
    with cB:
        st.download_button(
            "Download CSV (summary stats)",
            data=summ.to_csv(index=False).encode("utf-8"),
            file_name="summary_stats.csv",
            mime="text/csv"
        )

    # Forecast (raw + wide)
    with st.spinner("Building simple 24h forecast per variable…"):
        df_fc = simple_forecast(df, horizon_hours=24)
    if df_fc.empty:
        st.warning("Forecast could not be computed (too few points or invalid values).")
        fc_summary = pd.DataFrame()
        wide_fc = pd.DataFrame()
    else:
        st.subheader("Forecast (24h ahead, per variable)")
        for v in sorted(df_fc["variable"].unique()):
            dff = df_fc[df_fc["variable"] == v]
            base = df[df["variable"] == v]
            unitv = base["unit"].mode().iat[0] if not base.empty else ""
            figfc = px.line(dff, x="time", y="value", title=f"Forecast {v} ({unitv}) — 24 points")
            st.plotly_chart(figfc, use_container_width=True)

        st.subheader("Forecast summary statistics")
        g2 = df_fc.groupby("variable")["value"]
        fc_summary = pd.DataFrame({
            "mean": g2.mean(), "min": g2.min(), "max": g2.max(), "std": g2.std()
        }).reset_index()
        st.dataframe(fc_summary.sort_values("variable"), use_container_width=True)

        wide_fc = df_fc.pivot(index="time", columns="variable", values="value").reset_index()
        st.subheader("Forecast wide table (head)")
        st.dataframe(wide_fc.head(30), use_container_width=True)

    # PDF report
    meta = st.session_state.get("meta", {})
    with st.spinner("Building PDF report…"):
        pdf_bytes = build_pdf(
            df, summ,
            "AirCast — AQ + Satellite Report (RAW + Forecast Tables)",
            meta,
            sat_png_path=sat_png,
            sat_note="Raw satellite mosaic (no overlay)",
            forecast_summary=fc_summary if not fc_summary.empty else None,
            forecast_wide=wide_fc if not wide_fc.empty else None,
            sample_rows=df
        )
        st.session_state["aircast_pdf"] = pdf_bytes

# Persistent PDF download
if "aircast_pdf" in st.session_state:
    st.download_button(
        "Download PDF report",
        data=st.session_state["aircast_pdf"],
        file_name="aircast_report.pdf",
        mime="application/pdf",
        type="primary"
    )