# app/aircast.py
import io, math, itertools, tempfile, time, os
from typing import List, Dict, Any, Optional, Tuple
import requests
import pandas as pd
import plotly.express as px
from PIL import Image, ImageDraw
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# -------------------------
# Air Quality (Open-Meteo)
# -------------------------
OM_BASE = "https://air-quality-api.open-meteo.com/v1/air-quality"
DEFAULT_VARS = ["pm2_5","pm10","nitrogen_dioxide","ozone","sulphur_dioxide","carbon_monoxide"]

def fetch_hourly(lat: float, lon: float, past_days: int = 3,
                 variables: Optional[List[str]] = None,
                 timezone: str = "auto",
                 timeout: float = 20.0) -> Dict[str, Any]:
    vars_list = variables or DEFAULT_VARS
    params = {
        "latitude": lat,
        "longitude": lon,
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
        if var == "time": continue
        for t, v in zip(times, series):
            rows.append({"time": t, "variable": var, "value": v, "unit": units.get(var,"")})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.sort_values("time")
    return df

# -------------------------
# Satellite tiles (EOX primary; GIBS/ESRI fallback)
# -------------------------
UA = {"User-Agent": "Mozilla/5.0 StreamlitAQ/1.0"}
EOX_URL  = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg"
GIBS_URL = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief/default/{date}/GoogleMapsCompatible_Level{z}/{y}/{x}.jpg"
ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

def gibs_month() -> str:
    t = time.gmtime()
    return f"{t.tm_year:04d}-{t.tm_mon:02d}-01"

def lonlat_to_tile(lon: float, lat: float, z: int) -> Tuple[int,int]:
    import math as m
    lat_rad = m.radians(lat)
    n = 2.0 ** z
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - m.log(m.tan(lat_rad) + (1 / m.cos(lat_rad))) / m.pi) / 2.0 * n)
    return xtile, ytile

def http_get(url: str, timeout: float = 6.0, retries: int = 2, headers: Optional[Dict[str,str]] = None) -> Optional[bytes]:
    for _ in range(max(1, retries)):
        try:
            r = requests.get(url, timeout=timeout, headers=headers or UA)
            if r.status_code == 200 and r.content:
                return r.content
        except Exception:
            pass
    return None

def fetch_tile(provider: str, z: int, x: int, y: int) -> Optional[Image.Image]:
    if provider == "EOX":
        data = http_get(EOX_URL.format(z=z, x=x, y=y))
    elif provider == "GIBS":
        data = http_get(GIBS_URL.format(date=gibs_month(), z=z, x=x, y=y))
    else:
        data = http_get(ESRI_URL.format(z=z, x=x, y=y))
    if not data:
        return None
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None

def stitched_satellite(lon: float, lat: float, z: int = 8, grid: int = 3, size: int = 256,
                       prefer: str = "EOX") -> Tuple[Image.Image, str, int]:
    grid = max(1, grid)
    x0, y0 = lonlat_to_tile(lon, lat, z)
    half = grid // 2
    providers = [prefer] + [p for p in ["EOX","GIBS","ESRI"] if p != prefer]

    for prov in providers:
        canvas = Image.new("RGB", (size * grid, size * grid))
        ok = 0
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                x, y = x0 + dx, y0 + dy
                tile = fetch_tile(prov, z, x, y)
                if tile is None:
                    # Per-tile fallback chain
                    for fb in providers:
                        if fb == prov: 
                            continue
                        tile = fetch_tile(fb, z, x, y)
                        if tile is not None:
                            prov = f"{prov}→{fb}"
                            break
                if tile:
                    ok += 1
                    tile = tile.resize((size, size), Image.BILINEAR)
                    cx = (dx + half) * size; cy = (dy + half) * size
                    canvas.paste(tile, (cx, cy))
                else:
                    ph = Image.new("RGB", (size, size), (200, 200, 200))
                    cx = (dx + half) * size; cy = (dy + half) * size
                    canvas.paste(ph, (cx, cy))
        if ok > 0:
            return canvas, prov, ok

    # total failure: return gray
    return Image.new("RGB", (size * grid, size * grid), (200,200,200)), "NONE", 0

def test_one_tile(lon: float, lat: float, z: int) -> Dict[str, bool]:
    x, y = lonlat_to_tile(lon, lat, z)
    return {
        "EOX":  fetch_tile("EOX",  z, x, y)  is not None,
        "GIBS": fetch_tile("GIBS", z, x, y) is not None,
        "ESRI": fetch_tile("ESRI", z, x, y) is not None,
    }

# -------------------------
# Pollution overlay (simple radial heat)
# -------------------------
def color_for_value(v: float, vmin: float, vmax: float) -> Tuple[int,int,int]:
    """
    Green->Yellow->Red ramp; v normalized to [0,1].
    """
    import math
    if vmax <= vmin: vmax = vmin + 1.0
    t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
    # 0..0.5: green->yellow, 0.5..1: yellow->red
    if t < 0.5:
        a = t/0.5
        r,g,b = int(255*a), 255, 0
    else:
        a = (t-0.5)/0.5
        r,g,b = 255, int(255*(1-a)), 0
    return r,g,b

def make_radial_overlay(img_size: Tuple[int,int], base_color: Tuple[int,int,int], max_alpha: int = 140) -> Image.Image:
    """
    Create a radial alpha mask centered image; colored with base_color.
    """
    W, H = img_size
    overlay = Image.new("RGBA", (W, H), (0,0,0,0))
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    # radial gradient: multiple concentric circles
    steps = 40
    max_r = min(W, H) // 2
    for i in range(steps, 0, -1):
        r = int(max_r * i / steps)
        alpha = int(max_alpha * i / steps)
        bbox = (W//2 - r, H//2 - r, W//2 + r, H//2 + r)
        draw.ellipse(bbox, fill=alpha)
    # apply color
    color_layer = Image.new("RGBA", (W, H), base_color+(0,))
    overlay = Image.composite(color_layer, overlay, mask)
    return overlay

# -------------------------
# Summary + robust PDF (store bytes in session_state)
# -------------------------
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["variable","count","mean","median","min","max","std"])
    g = df.groupby("variable")["value"]
    return pd.DataFrame({
        "count": g.count(), "mean": g.mean(), "median": g.median(),
        "min": g.min(), "max": g.max(), "std": g.std(),
    }).reset_index()

def build_pdf_bytes(df: pd.DataFrame, summary: pd.DataFrame, title: str, meta: str,
                    sat_png_path: Optional[str], sat_note: Optional[str]) -> bytes:
    # write to a temp file then read bytes
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    path = tmp.name
    tmp.close()

    c = canvas.Canvas(path, pagesize=A4)
    W, H = A4; margin = 1.5*cm; y = H - margin
    c.setFont("Helvetica-Bold", 14); c.drawString(margin, y, title); y -= 0.8*cm
    c.setFont("Helvetica", 9); c.drawString(margin, y, meta); y -= 0.5*cm
    if sat_note:
        c.setFont("Helvetica-Oblique", 8); c.drawString(margin, y, sat_note); y -= 0.5*cm

    if sat_png_path and os.path.exists(sat_png_path):
        try:
            img = ImageReader(sat_png_path)
            max_w = W - 2*margin; max_h = 7*cm
            c.drawImage(img, margin, y - max_h, width=max_w, height=max_h, preserveAspectRatio=True, anchor='n')
            y -= (max_h + 0.5*cm)
        except Exception:
            c.setFont("Helvetica-Oblique", 9); c.drawString(margin, y, "(Satellite image failed)"); y -= 0.6*cm

    c.setFont("Helvetica-Bold", 11); c.drawString(margin, y, "Summary by variable"); y -= 0.6*cm
    c.setFont("Helvetica-Bold", 9)
    headers = ["variable","count","mean","median","min","max","std"]
    xs = [margin, margin+4*cm, margin+6*cm, margin+8*cm, margin+10*cm, margin+12*cm, margin+14*cm]
    for h, x in zip(headers, xs): c.drawString(x, y, h.upper())
    y -= 0.5*cm; c.setFont("Helvetica", 9)
    for _, r in summary.iterrows():
        vals = [str(r[h]) if h=="variable" else f"{r[h]:.2f}" for h in headers]
        for val, x in zip(vals, xs): c.drawString(x, y, val)
        y -= 0.4*cm
        if y < 2*cm:
            c.showPage(); y = H - margin
            c.setFont("Helvetica-Bold", 11); c.drawString(margin, y, "Summary (cont.)"); y -= 0.6*cm
            c.setFont("Helvetica", 9)

    c.showPage(); y = H - margin
    c.setFont("Helvetica-Bold", 11); c.drawString(margin, y, "Sample (first 40 rows)"); y -= 0.6*cm
    c.setFont("Helvetica", 8)
    for _, r in df.head(40).iterrows():
        unit = r.get("unit","") if pd.notna(r.get("unit","")) else ""
        line = f"{r['time']} | {r['variable']} = {r['value']} {unit}"
        c.drawString(margin, y, line); y -= 0.35*cm
        if y < 2*cm:
            c.showPage(); y = H - margin; c.setFont("Helvetica", 8)
    c.save()

    with open(path, "rb") as fh:
        data = fh.read()
    try:
        os.remove(path)
    except Exception:
        pass
    return data

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AeroVision — AQ + Satellite", layout="wide")
st.title("AeroVision — Online AQ (Open-Meteo) + Satellite (EOX) with Pollution Overlay")

with st.sidebar:
    st.header("Controls")
    c1, c2 = st.columns(2)
    lat = c1.text_input("Latitude", "12.9716")
    lon = c2.text_input("Longitude", "77.5946")
    past_days = st.slider("Past days (1..7)", 1, 7, 3)
    vars_sel = st.multiselect("Variables", DEFAULT_VARS, default=["pm2_5","pm10","nitrogen_dioxide","ozone"])

    st.divider(); st.caption("Satellite & Overlay")
    sat_zoom = st.slider("Zoom (6–15)", 6, 15, 12)
    sat_grid = st.slider("Grid size (tiles/side)", 1, 5, 3)
    sat_tile_px = st.radio("Tile px", [256, 512], index=0)
    overlay_param = st.selectbox("Overlay pollutant (color)", ["nitrogen_dioxide","pm2_5","pm10","ozone"], index=0)
    overlay_strength = st.slider("Overlay strength (alpha)", 40, 180, 120)

colA, colB = st.columns([1,1])
if colA.button("Test satellite connectivity"):
    try:
        lat_f, lon_f = float(lat), float(lon)
        res = test_one_tile(lon_f, lat_f, z=int(sat_zoom))
        st.write({"EOX": res["EOX"], "GIBS": res["GIBS"], "ESRI": res["ESRI"]})
    except:
        st.error("Invalid coordinates.")

go = colB.button("Fetch & render", type="primary")

if go:
    try:
        lat_f, lon_f = float(lat), float(lon)
    except:
        st.error("Invalid coordinates. Example: 12.9716, 77.5946")
        st.stop()

    # AQ
    with st.spinner("Calling Open-Meteo (hourly AQ)…"):
        try:
            payload = fetch_hourly(lat_f, lon_f, past_days=past_days, variables=vars_sel or DEFAULT_VARS)
            df = om_to_dataframe(payload)
        except Exception as e:
            st.error(f"AQ fetch failed: {e}")
            st.stop()
    if df.empty:
        st.warning("No AQ data. Try different coordinates or fewer days.")
        st.stop()

    st.success(f"AQ OK — {len(df)} hourly points across {df['variable'].nunique()} variables.")

    # Satellite (EOX preferred)
    with st.spinner("Fetching satellite mosaic (EOX)…"):
        sat_img, provider_used, ok = stitched_satellite(
            lon=lon_f, lat=lat_f, z=int(sat_zoom), grid=int(sat_grid), size=int(sat_tile_px), prefer="EOX"
        )
        total = sat_grid * sat_grid
        if ok == 0:
            st.warning("Satellite tiles failed. If behind firewall, allow tiles.maps.eox.at")
        else:
            st.info(f"Satellite provider: {provider_used} | tiles ok {ok}/{total}")

    # Overlay based on NO2/PM values (last 24h mean)
    dfo = df[df["variable"] == overlay_param]
    overlay_png_path = None
    overlay_note = None
    if not dfo.empty:
        last24 = dfo[dfo["time"] >= (dfo["time"].max() - pd.Timedelta(hours=24))]
        val = float(last24["value"].mean()) if not last24.empty else float(dfo["value"].mean())
        # naive expected ranges (you can tweak):
        ranges = {
            "nitrogen_dioxide": (0, 80),   # µg/m3 typical urban
            "pm2_5": (0, 100),
            "pm10": (0, 150),
            "ozone": (0, 180)
        }
        vmin, vmax = ranges.get(overlay_param, (0, 100))
        color = color_for_value(val, vmin, vmax)
        heat = make_radial_overlay(sat_img.size, color, max_alpha=int(overlay_strength))
        sat_img_rgba = sat_img.convert("RGBA")
        sat_img_rgba.alpha_composite(heat)
        sat_img = sat_img_rgba.convert("RGB")
        overlay_note = f"{overlay_param} 24h mean ≈ {val:.1f} (range {vmin}-{vmax})"
    else:
        overlay_note = f"No values for {overlay_param}; showing raw satellite."

    st.subheader("Satellite with pollution overlay")
    st.caption(overlay_note)
    st.image(sat_img, use_container_width=True)

    # Charts
    st.subheader("Hourly time-series")
    for v in sorted(df["variable"].unique()):
        dff = df[df["variable"] == v]
        unit = dff["unit"].mode().iat[0] if not dff["unit"].empty else ""
        fig = px.line(dff, x="time", y="value", title=f"{v} ({unit}) — {len(dff)} points")
        st.plotly_chart(fig, use_container_width=True)

    # Summary
    st.subheader("Summary statistics")
    summ = summarize(df)
    st.dataframe(summ, use_container_width=True)

    # Prepare assets for PDF and expose a stable download button
    sat_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    sat_img.save(sat_tmp.name, format="PNG")
    sat_png_path = sat_tmp.name

    meta = (f"lat,lon={lat_f:.4f},{lon_f:.4f} | past_days={past_days} | "
            f"vars={','.join(vars_sel or DEFAULT_VARS)} | overlay={overlay_param} | "
            f"zoom={sat_zoom}, grid={sat_grid}x{sat_grid}")

    pdf_bytes = build_pdf_bytes(df, summ, "AirCast — AQ + Satellite Report", meta,
                                sat_png_path=sat_png_path, sat_note=overlay_note)

    # Keep in session so it persists across reruns
    st.session_state["aircast_pdf"] = pdf_bytes

# Show download button whenever PDF is present (prevents “redirect to home” issue)
if "aircast_pdf" in st.session_state:
    st.download_button(
        "Download PDF report",
        data=st.session_state["aircast_pdf"],
        file_name="aircast_report.pdf",
        mime="application/pdf",
        type="primary"
    )
