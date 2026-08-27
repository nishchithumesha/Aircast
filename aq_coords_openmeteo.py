# app/aq_coords_openmeteo.py
import io, math, itertools, tempfile, time
from typing import List, Dict, Any, Optional, Tuple
import requests
import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st

# ====== PDF helpers ======
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# =========================
# 1) Air Quality (Open-Meteo) – no key
# =========================
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

# =========================
# 2) Satellite tiles with fallback + retries
#    Providers:
#      A) ESRI World Imagery (no key)
#      B) NASA GIBS BlueMarble (monthly, stable)
# =========================
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StreamlitAQ/1.0",
      "Referer": "https://www.arcgis.com/"}

ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
# BlueMarble monthly: pick first of this month. JPG tiles.
def gibs_month_path() -> str:
    t = time.gmtime()
    return f"{t.tm_year:04d}-{t.tm_mon:02d}-01"

GIBS_URL = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief/default/{date}/GoogleMapsCompatible_Level{z}/{y}/{x}.jpg"

def lonlat_to_tile(lon: float, lat: float, z: int) -> Tuple[int,int]:
    lat_rad = math.radians(lat)
    n = 2.0 ** z
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile

def http_get(url: str, timeout: float = 8.0, retries: int = 2, headers: Optional[Dict[str,str]] = None) -> Optional[bytes]:
    last_err = None
    for _ in range(max(1,retries)):
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            if r.status_code == 200 and r.content:
                return r.content
        except Exception as e:
            last_err = e
    return None

def fetch_tile_esri(z: int, x: int, y: int) -> Optional[Image.Image]:
    data = http_get(ESRI_URL.format(z=z,x=x,y=y), headers=UA)
    if not data: return None
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None

def fetch_tile_gibs(z: int, x: int, y: int, date_str: Optional[str] = None) -> Optional[Image.Image]:
    d = date_str or gibs_month_path()
    data = http_get(GIBS_URL.format(date=d, z=z, x=x, y=y), headers={"User-Agent": UA["User-Agent"]})
    if not data: return None
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None

def stitched_satellite(lon: float, lat: float, z: int = 12, grid: int = 3, size: int = 256,
                       provider: str = "ESRI") -> Tuple[Image.Image, int]:
    """
    Returns (image, success_count). Gray placeholders where tiles fail.
    provider: "ESRI" or "GIBS"
    """
    grid = max(1, grid)
    x0, y0 = lonlat_to_tile(lon, lat, z)
    half = grid // 2
    canvas = Image.new("RGB", (size * grid, size * grid))
    ok = 0
    for dy, dx in itertools.product(range(-half, half + 1), range(-half, half + 1)):
        x, y = x0 + dx, y0 + dy
        tile = None
        if provider == "ESRI":
            tile = fetch_tile_esri(z, x, y)
        else:
            tile = fetch_tile_gibs(z, x, y)
        if tile is None and provider == "ESRI":
            # ESRI failed → try GIBS per tile
            tile = fetch_tile_gibs(z, x, y)
        if tile:
            ok += 1
            tile = tile.resize((size, size), Image.BILINEAR)
            cx = (dx + half) * size; cy = (dy + half) * size
            canvas.paste(tile, (cx, cy))
        else:
            ph = Image.new("RGB", (size, size), (200, 200, 200))
            cx = (dx + half) * size; cy = (dy + half) * size
            canvas.paste(ph, (cx, cy))
    return canvas, ok

# =========================
# 3) Stats + robust PDF
# =========================
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["variable","count","mean","median","min","max","std"])
    g = df.groupby("variable")["value"]
    out = pd.DataFrame({
        "count": g.count(), "mean": g.mean(), "median": g.median(),
        "min": g.min(), "max": g.max(), "std": g.std(),
    }).reset_index()
    return out

def export_pdf(df: pd.DataFrame, summary: pd.DataFrame, path: str, title: str, meta: str,
               sat_png_path: Optional[str] = None, sat_note: Optional[str] = None):
    c = canvas.Canvas(path, pagesize=A4)
    W, H = A4; margin = 1.5*cm; y = H - margin

    c.setFont("Helvetica-Bold", 14); c.drawString(margin, y, title); y -= 0.8*cm
    c.setFont("Helvetica", 9); c.drawString(margin, y, meta); y -= 0.5*cm
    if sat_note:
        c.setFont("Helvetica-Oblique", 8); c.drawString(margin, y, sat_note); y -= 0.5*cm

    # Satellite (optional, but we still generate PDF without it)
    if sat_png_path:
        try:
            img = ImageReader(sat_png_path)
            max_w = W - 2*margin; max_h = 7*cm
            c.drawImage(img, margin, y - max_h, width=max_w, height=max_h, preserveAspectRatio=True, anchor='n')
            y -= (max_h + 0.5*cm)
        except Exception:
            c.setFont("Helvetica-Oblique", 9)
            c.drawString(margin, y, "(Satellite image failed to render)"); y -= 0.6*cm

    # Summary table
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

# =========================
# 4) Streamlit UI
# =========================
st.set_page_config(page_title="AirCast — AQ + Satellite (Fast)", layout="wide")
st.title("AirCast — Online AQ (Open-Meteo) + Satellite (ESRI/GIBS)")

with st.sidebar:
    st.header("Controls")
    c1, c2 = st.columns(2)
    lat = c1.text_input("Latitude", "12.9716")
    lon = c2.text_input("Longitude", "77.5946")
    past_days = st.slider("Past days (1..7)", 1, 7, 3)
    var_choices = DEFAULT_VARS
    vars_sel = st.multiselect("Variables", var_choices, default=["pm2_5","pm10","nitrogen_dioxide","ozone"])

    st.divider(); st.caption("Satellite")
    provider = st.radio("Provider", ["ESRI (World Imagery)","NASA GIBS (BlueMarble)"], index=0)
    sat_zoom = st.slider("Zoom (3–15)", 3, 15, 12)
    sat_grid = st.slider("Grid size (tiles/side)", 1, 5, 3)
    sat_tile_px = st.radio("Tile px", [256, 512], index=0)

btn = st.button("Fetch data", type="primary")

if btn:
    # Parse coords
    try:
        lat_f, lon_f = float(lat), float(lon)
    except:
        st.error("Invalid coordinates. Example: 12.9716, 77.5946")
        st.stop()

    # ===== AQ fetch =====
    with st.spinner("Calling Open-Meteo (hourly AQ)..."):
        try:
            payload = fetch_hourly(lat_f, lon_f, past_days=past_days, variables=vars_sel or DEFAULT_VARS)
            df = om_to_dataframe(payload)
        except Exception as e:
            st.error(f"AQ fetch failed: {e}")
            st.stop()

    if df.empty:
        st.warning("No AQ data returned. Try different coordinates or fewer days.")
        st.stop()

    st.success(f"AQ OK — {len(df)} hourly points across {df['variable'].nunique()} variables.")

    # ===== Satellite mosaic =====
    sat_png_path, sat_note = None, None
    with st.spinner("Fetching satellite mosaic..."):
        prov_key = "ESRI" if provider.startswith("ESRI") else "GIBS"
        sat_img, ok = stitched_satellite(
            lon=lon_f, lat=lat_f, z=int(sat_zoom), grid=int(sat_grid), size=int(sat_tile_px),
            provider=prov_key
        )
        if ok == 0:
            sat_note = f"Satellite tiles failed for {provider}. Check firewall/network; tried {sat_grid*sat_grid} tiles."
            st.warning(sat_note)
        else:
            st.subheader(f"Satellite imagery — {provider}  (ok tiles: {ok}/{sat_grid*sat_grid})")
            st.image(sat_img, use_container_width=True,
                     caption=f"Center: {lat_f:.4f},{lon_f:.4f} | z={sat_zoom}, grid={sat_grid}×{sat_grid}")
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            sat_img.save(tmp.name, format="PNG"); sat_png_path = tmp.name

    # ===== Charts =====
    st.subheader("Hourly time-series")
    for v in sorted(df["variable"].unique()):
        dff = df[df["variable"] == v]
        unit = dff["unit"].mode().iat[0] if not dff["unit"].empty else ""
        fig = px.line(dff, x="time", y="value", title=f"{v} ({unit}) — {len(dff)} points")
        st.plotly_chart(fig, use_container_width=True)

    # ===== Summary + downloads =====
    st.subheader("Summary statistics")
    summ = summarize(df)
    st.dataframe(summ, use_container_width=True)

    cA, cB, cC = st.columns(3)
    with cA:
        st.download_button("Download CSV (raw)", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="openmeteo_raw.csv", mime="text/csv")
    with cB:
        st.download_button("Download CSV (summary)", data=summ.to_csv(index=False).encode("utf-8"),
                           file_name="openmeteo_summary.csv", mime="text/csv")
    with cC:
        if st.button("Generate PDF report"):
            tmppdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            meta = (f"lat,lon={lat_f:.4f},{lon_f:.4f} | past_days={past_days} | "
                    f"vars={','.join(vars_sel or DEFAULT_VARS)} | provider={provider} | "
                    f"z={sat_zoom}, grid={sat_grid}x{sat_grid}")
            try:
                export_pdf(df, summ, tmppdf.name, "AirCast — Open-Meteo AQ Report",
                           meta, sat_png_path=sat_png_path, sat_note=sat_note)
                with open(tmppdf.name, "rb") as f:
                    st.download_button("Download PDF", data=f.read(),
                                       file_name="openmeteo_report.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
