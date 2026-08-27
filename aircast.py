# app/aircast.py
import io, os, time, math, itertools, tempfile, random
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd
import plotly.express as px
from PIL import Image, ImageDraw, ImageFilter, ImageChops
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# =========================
# 1) Open-Meteo Air Quality
# =========================
OM_BASE = "https://air-quality-api.open-meteo.com/v1/air-quality"
DEFAULT_VARS = ["pm2_5","pm10","nitrogen_dioxide","ozone","sulphur_dioxide","carbon_monoxide"]

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
        df = df.sort_values("time")
    return df

# =========================
# 2) Satellite tiles (EOX→GIBS→ESRI)
# =========================
UA = {"User-Agent": "Mozilla/5.0 StreamlitAQ/1.0"}
EOX_URL  = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg"
def gibs_month() -> str:
    t = time.gmtime()
    return f"{t.tm_year:04d}-{t.tm_mon:02d}-01"
GIBS_URL = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief/default/{date}/GoogleMapsCompatible_Level{z}/{y}/{x}.jpg"
ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

def lonlat_to_tile(lon: float, lat: float, z: int) -> Tuple[int,int]:
    lat_rad = math.radians(lat)
    n = 2.0 ** z
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
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

def stitched_satellite(lon: float, lat: float, z: int = 12, grid: int = 3, size: int = 256,
                       prefer: str = "EOX") -> Tuple[Image.Image, str, int]:
    grid = max(1, grid)
    x0, y0 = lonlat_to_tile(lon, lat, z)
    half = grid // 2
    providers = [prefer] + [p for p in ["EOX","GIBS","ESRI"] if p != prefer]

    canvas_img = Image.new("RGB", (size * grid, size * grid))
    ok = 0
    provider_used = prefer
    for dy, dx in itertools.product(range(-half, half + 1), range(-half, half + 1)):
        x, y = x0 + dx, y0 + dy
        tile, chosen = None, None
        for prov in providers:
            tile = fetch_tile(prov, z, x, y)
            if tile is not None:
                chosen = prov
                break
        if tile:
            ok += 1
            tile = tile.resize((size, size), Image.BILINEAR)
            cx = (dx + half) * size; cy = (dy + half) * size
            canvas_img.paste(tile, (cx, cy))
            provider_used = chosen or provider_used
        else:
            ph = Image.new("RGB", (size, size), (200, 200, 200))
            cx = (dx + half) * size; cy = (dy + half) * size
            canvas_img.paste(ph, (cx, cy))
    return canvas_img, provider_used, ok

def test_one_tile(lon: float, lat: float, z: int) -> Dict[str, bool]:
    x, y = lonlat_to_tile(lon, lat, z)
    return {
        "EOX":  fetch_tile("EOX",  z, x, y)  is not None,
        "GIBS": fetch_tile("GIBS", z, x, y) is not None,
        "ESRI": fetch_tile("ESRI", z, x, y) is not None,
    }

# =========================
# 3) S5P-style palette + overlay + legend
# =========================
def _hex2rgb(h: str) -> Tuple[int,int,int]:
    h = h.lstrip('#'); return tuple(int(h[i:i+2],16) for i in (0,2,4))

THERMAL_PALETTES = {
    "s5p":    ["#00004c","#002c88","#005bb7","#00a6c7","#36d3c4","#7be07a","#d9e14b","#f6c33b","#f6952b","#ec4b1b"],
    "plasma": ["#0d0887","#5b02a3","#9a179b","#cb4679","#ed7953","#fb9f3a","#fdca26","#f0f921"],
    "turbo":  ["#2318c9","#3b4ded","#4897fb","#52c6ee","#67e7c2","#8ef193","#b8f15f","#e3d33b","#f6952b","#f13d1b"],
    "inferno":["#000004","#1f0c48","#55106d","#88226a","#b9375a","#dd513a","#f57d15","#fdb42f","#f9dd3a","#fcffa4"],
}

def colormap_color(t: float, name: str="s5p") -> Tuple[int,int,int]:
    ramp = THERMAL_PALETTES.get(name, THERMAL_PALETTES["s5p"])
    if t <= 0: return _hex2rgb(ramp[0])
    if t >= 1: return _hex2rgb(ramp[-1])
    pos = t*(len(ramp)-1)
    i, frac = int(pos), pos-int(pos)
    c1, c2 = _hex2rgb(ramp[i]), _hex2rgb(ramp[i+1])
    r = int(c1[0] + (c2[0]-c1[0])*frac)
    g = int(c1[1] + (c2[1]-c1[1])*frac)
    b = int(c1[2] + (c2[2]-c1[2])*frac)
    return (r,g,b)

def draw_colorbar(palette_name: str, vmin: float, vmax: float, unit: str, w=420, h=18) -> Image.Image:
    bar = Image.new("RGB", (w, h+18), (255,255,255))
    drw = ImageDraw.Draw(bar)
    for x in range(w):
        t = x / (w - 1)
        drw.line([(x, 0), (x, h - 1)], fill=colormap_color(t, palette_name), width=1)
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    for t in ticks:
        x = int(t*(w-1))
        drw.line([(x, h), (x, h+4)], fill=(0,0,0))
        val = vmin + t*(vmax-vmin)
        label = f"{val:.0f}"
        drw.text((max(0, x - len(label)*3), h+6), label, fill=(0,0,0))
    if unit:
        drw.text((w-40, h+6), unit, fill=(0,0,0))
    return bar

def make_radial_overlay(img_size: Tuple[int,int], base_color: Tuple[int,int,int], max_alpha: int = 140,
                        center: Optional[Tuple[int,int]] = None, radius_frac: float = 0.5) -> Image.Image:
    W, H = img_size
    cx, cy = center if center else (W//2, H//2)
    overlay = Image.new("RGBA", (W, H), (0,0,0,0))
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    steps = 40
    max_r = int(min(W, H) * radius_frac)
    for i in range(steps, 0, -1):
        r = int(max_r * i / steps)
        alpha = int(max_alpha * i / steps)
        bbox = (cx - r, cy - r, cx + r, cy + r)
        draw.ellipse(bbox, fill=alpha)
    color_layer = Image.new("RGBA", (W, H), base_color+(0,))
    return Image.composite(color_layer, overlay, mask)

# =========================
# 4) Summary + PDF
# =========================
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
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    path = tmp.name; tmp.close()

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
        c.drawString(margin, y, f"{r['time']} | {r['variable']} = {r['value']} {unit}")
        y -= 0.35*cm
        if y < 2*cm:
            c.showPage(); y = H - margin; c.setFont("Helvetica", 8)
    c.save()

    with open(path, "rb") as fh:
        data = fh.read()
    try: os.remove(path)
    except Exception: pass
    return data

# =========================
# 5) Streamlit UI
# =========================
st.set_page_config(page_title="AeroVision — AQ + Satellite (Sentinel-5P style)", layout="wide")
st.title("AeroVision — Online AQ + Satellite Mosaic — Sentinel-5P style")

with st.sidebar:
    st.header("Controls")
    c1, c2 = st.columns(2)
    lat = c1.text_input("Latitude", "12.9716")
    lon = c2.text_input("Longitude", "77.5946")
    past_days = st.slider("Past days (1–7)", 1, 7, 3)
    vars_sel = st.multiselect("Variables", DEFAULT_VARS, default=["pm2_5","pm10","nitrogen_dioxide","ozone","sulphur_dioxide"])

    st.divider(); st.caption("Satellite")
    sat_zoom = st.slider("Zoom (6–15)", 6, 15, 12)
    sat_grid = st.slider("Grid size (tiles/side)", 1, 5, 3)
    sat_tile_px = st.radio("Tile size (px)", [256, 512], index=0)

    # REMOVED "Main overlay" WIDGETS

# --- Hardcode defaults for the removed sidebar widgets
overlay_param = "nitrogen_dioxide"
thermal_palette = "s5p"
# Using a higher alpha strength for the visuals to stand out against the darkened background
overlay_strength = 180 
overlay_radius = 0.5

colA, colB = st.columns(2)
if colA.button("Test satellite connectivity"):
    try:
        lat_f, lon_f = float(lat), float(lon)
        res = test_one_tile(lon_f, lat_f, z=int(sat_zoom))
        st.write(res)
        st.info("At least one True means that provider works on your network.")
    except:
        st.error("Invalid coordinates.")

go = colB.button("Fetch & render", type="primary")

if go:
    # Parse coords
    try:
        lat_f, lon_f = float(lat), float(lon)
    except:
        st.error("Invalid coordinates. Example: 12.9716, 77.5946")
        st.stop()

    # ---- AQ fetch
    with st.spinner("Calling Open-Meteo (hourly AQ)…"):
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

    # ---- Satellite mosaic
    with st.spinner("Fetching satellite mosaic (EOX→GIBS→ESRI)…"):
        sat_img, provider_used, ok = stitched_satellite(
            lon=lon_f, lat=lat_f, z=int(sat_zoom), grid=int(sat_grid), size=int(sat_tile_px), prefer="EOX"
        )
    total = sat_grid * sat_grid
    if ok == 0:
        st.warning("Satellite tiles failed from all providers. If behind a firewall, allow: tiles.maps.eox.at, gibs.earthdata.nasa.gov, server.arcgisonline.com")
    else:
        st.info(f"Satellite provider used: {provider_used} — tiles OK {ok}/{total}")

    # ---- Logic for PDF/Variables (retained)
    ranges = {
        "nitrogen_dioxide": (0, 80),  # µg/m³ (Open-Meteo units)
        "pm2_5": (0, 100),
        "pm10": (0, 150),
        "ozone": (0, 180),
        "sulphur_dioxide": (0, 70),
    }
    units = {k: "µg/m³" for k in ranges.keys()}

    dfo = df[df["variable"] == overlay_param]
    vmin, vmax = ranges.get(overlay_param, (0, 100))
    unit = units.get(overlay_param, "")
    val = 0.0

    if not dfo.empty:
        last24 = dfo[dfo["time"] >= (dfo["time"].max() - pd.Timedelta(hours=24))]
        val = float(last24["value"].mean()) if not last24.empty else float(dfo["value"].mean())
        overlay_note = f"Overlay data derived from: {overlay_param} 24h mean ≈ {val:.1f} {unit} (range {vmin}-{vmax}), palette: {thermal_palette}"
    else:
        overlay_note = f"No recent values for {overlay_param}; showing raw satellite."

    # --- Display the RAW satellite mosaic as a base for clarity ---
    st.subheader("Raw Satellite Mosaic")
    st.image(sat_img, use_container_width=True)
    
    raw_sat_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    sat_img.save(raw_sat_png.name, format="PNG")
    with open(raw_sat_png.name, "rb") as fh:
        st.download_button("Download Raw Satellite PNG", data=fh.read(),
                           file_name="sat_raw_mosaic.png", mime="image/png")

    # ---- Charts
    st.subheader("Hourly time-series")
    for v in sorted(df["variable"].unique()):
        dff = df[df["variable"] == v]
        unitv = dff["unit"].mode().iat[0] if not dff["unit"].empty else ""
        fig = px.line(dff, x="time", y="value", title=f"{v} ({unitv}) — {len(dff)} points")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Summary + CSVs
    st.subheader("Summary statistics")
    summ = summarize(df)
    st.dataframe(summ, use_container_width=True)

    cA, cB = st.columns(2)
    with cA:
        st.download_button("Download CSV (raw)", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="openmeteo_raw.csv", mime="text/csv")
    with cB:
        st.download_button("Download CSV (summary)", data=summ.to_csv(index=False).encode("utf-8"),
                           file_name="openmeteo_summary.csv", mime="text/csv")

    # ---- PDF (using raw satellite image for the map, as the main overlay is removed)
    sat_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    sat_img.save(sat_tmp.name, format="PNG")
    meta = (
        f"lat,lon={lat_f:.4f},{lon_f:.4f} | past_days={past_days} | "
        f"vars={','.join(vars_sel or DEFAULT_VARS)} | overlay_default={overlay_param} | "
        f"zoom={sat_zoom}, grid={sat_grid}x{sat_grid} | provider={provider_used}"
    )
    pdf_bytes = build_pdf_bytes(df, summ, "AirCast — AQ + Satellite Report",
                                meta, sat_png_path=sat_tmp.name, sat_note=overlay_note)
    st.session_state["aircast_pdf"] = pdf_bytes

    # ---- Data-driven gallery: NO2, SO2, O3 (with simulated fallback)
    st.divider()
    st.subheader("Sentinel-5P-style pollutant visuals — data-driven gallery")

    POLLUTANTS = [
        ("nitrogen_dioxide", "NO₂",  "s5p",    (0, 80),  "µg/m³", 111),
        ("sulphur_dioxide",  "SO₂",  "plasma", (0, 70),  "µg/m³", 222),
        ("ozone",            "O₃",   "turbo",  (0, 180), "µg/m³", 333),
    ]
    
    # --- Function to darken the base image ---
    def darken_base(img_rgba: Image.Image, factor: float = 0.2) -> Image.Image:
        r, g, b, a = img_rgba.split()
        r = r.point(lambda p: int(p * factor))
        g = g.point(lambda p: int(p * factor))
        b = b.point(lambda p: int(p * factor))
        # Ensure the background is dark-blue/black instead of just dark-gray
        dark_overlay = Image.new("RGBA", img_rgba.size, (0, 0, 48, 255))
        darkened_rgb = Image.merge("RGB", (r, g, b))
        darkened_rgba = darkened_rgb.convert("RGBA")
        return Image.alpha_composite(dark_overlay, darkened_rgba)


    def plume_from_stats(base_rgba: Image.Image,
                         df: pd.DataFrame,
                         param: str,
                         palette: str,
                         vrange: Tuple[float,float],
                         seed: int,
                         alpha: int = 200) -> Tuple[Image.Image, str]:
        
        # --- NEW: Darken the base image to match S5P style ---
        dark_base_rgba = darken_base(base_rgba)
        frame = dark_base_rgba.copy()
        
        W, H = base_rgba.size
        dfo = df[df["variable"] == param]
        vmin, vmax = vrange
        rng = random.Random(seed)

        # --- Live Data Plume ---
        if not dfo.empty:
            last24 = dfo[dfo["time"] >= (dfo["time"].max() - pd.Timedelta(hours=24))]
            val = float(last24["value"].mean()) if not last24.empty else float(dfo["value"].mean())
            t = 0.0 if vmax <= vmin else max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
            color = colormap_color(t, palette)

            # --- Plume generation logic (enhanced from previous step) ---
            wind_len = rng.randint(25, 50) 
            ang = rng.uniform(-math.pi/4, math.pi/4) 
            vx, vy = int(wind_len*math.cos(ang)), int(wind_len*math.sin(ang))

            n_blobs = rng.randint(2, 5) 
            for _ in range(n_blobs):
                cx = rng.randint(int(W*0.20), int(W*0.80))
                cy = rng.randint(int(H*0.20), int(H*0.80))
                rfrac = rng.uniform(0.30, 0.60)
                # Use a high alpha to make the plume dominant
                amax = int(alpha * rng.uniform(0.8, 1.0)) 
                blob = make_radial_overlay((W, H), color, max_alpha=amax, center=(cx, cy), radius_frac=rfrac)

                copies = rng.randint(5, 8) 
                accum = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                for j in range(copies):
                    fade = int(amax * (1.0 - (j+1)/copies) * 0.9)
                    buf = Image.new("RGBA", (W, H), (0,0,0,0))
                    buf.alpha_composite(blob)
                    r_, g_, b_, a_ = buf.split()
                    a_ = a_.point(lambda p: min(255, int(p * (fade/255.0))))
                    buf = Image.merge("RGBA", (r_, g_, b_, a_))
                    dx = int(vx * (j+1)/copies); dy = int(vy * (j+1)/copies)
                    accum.alpha_composite(buf, (dx, dy))
                blob.alpha_composite(accum)
                frame = Image.alpha_composite(frame, blob) # Compose over the dark base
            
            frame = frame.filter(ImageFilter.GaussianBlur(radius=1.5))
            caption = f"{param} 24h mean ≈ {val:.1f} {unit}"
            return frame, caption

        # --- Fallback Simulated Plume (Enhanced) ---
        def _value_noise(w, h, cell=64, seed=42):
            rnd = random.Random(seed)
            gx, gy = max(2, w // cell), max(2, h // cell)
            grid = [[rnd.random() for _ in range(gx+1)] for __ in range(gy+1)]
            img = Image.new("L", (w, h), 0); px = img.load()
            for y in range(h):
                yf = y / cell; y0 = int(yf); yt = yf - y0; y1 = min(y0+1, gy)
                for x in range(w):
                    xf = x / cell; x0 = int(xf); xt = xf - x0; x1 = min(x0+1, gx)
                    v00 = grid[y0][x0]; v10 = grid[y0][x1]; v01 = grid[y1][x0]; v11 = grid[y1][x1]
                    v0 = v00*(1-xt) + v10*xt; v1 = v01*(1-xt) + v11*xt
                    px[x,y] = int(255*(v0*(1-yt) + v1*yt))
            return img.filter(ImageFilter.GaussianBlur(radius=5)) 

        noise = _value_noise(W, H, cell=random.choice([40,56,72]), seed=seed) 
        low_clip, gain = random.uniform(0.40, 0.60), random.uniform(1.5, 2.0)
        alphaL = noise.point(lambda p: max(0, min(255, int((p/255-low_clip)*gain*255))))
        wind = random.randint(25, 50); ang = random.uniform(-math.pi/4, math.pi/4)
        vx, vy = int(wind*math.cos(ang)), int(wind*math.sin(ang))

        adv = Image.new("L", (W, H), 0)
        for j in range(1, random.randint(6,9)+1): 
            f = int(255 * (1 - j/(j+1)) * 0.9) 
            sh = Image.new("L", (W, H), 0)
            sh.paste(alphaL, (int(vx*j/(j+1)), int(vy*j/(j+1))))
            adv = ImageChops.lighter(adv, sh.point(lambda p: int(p*f/255)))

        a_chan = adv.point(lambda p: int(p * (alpha/255.0)))
        r_chan = Image.new("L", (W, H), 0)
        g_chan = Image.new("L", (W, H), 0)
        b_chan = Image.new("L", (W, H), 0)
        apx = a_chan.load(); rpx, gpx, bpx = r_chan.load(), g_chan.load(), b_chan.load()
        for y in range(H):
            for x in range(W):
                t = apx[x,y] / 255.0
                r,g,b = colormap_color(t, palette)
                rpx[x,y], gpx[x,y], bpx[x,y] = r,g,b
        colorized = Image.merge("RGBA", (r_chan, g_chan, b_chan, a_chan))
        frame = Image.alpha_composite(frame, colorized)
        frame = frame.filter(ImageFilter.GaussianBlur(radius=1.5)) 
        caption = "Simulated plume (no live data)"
        return frame, caption

    base_rgba = sat_img.convert("RGBA")
    cols = st.columns(3)
    for (param, short, pal, vrange, unit, seed), col in zip(POLLUTANTS, cols):
        fimg, cap = plume_from_stats(base_rgba, df, param, pal, vrange, seed, alpha=int(overlay_strength))
        col.image(fimg.convert("RGB"), use_container_width=True,
                  caption=f"**{short}** ({param.replace('_',' ')}): {cap}")

    # Add colorbar for the Gallery visuals
    st.divider()
    st.caption("Color Scale Reference (used in Gallery above):")
    c1, c2, c3 = st.columns(3)
    
    for (param, short, pal, vrange, unit, seed), col in zip(POLLUTANTS, [c1, c2, c3]):
        vmin, vmax = vrange
        legend = draw_colorbar(pal, vmin=vmin, vmax=vmax, unit=unit, w=300, h=15)
        col.image(legend, caption=f"**{short}** ({pal} palette)", use_container_width=True)


# Persistent PDF download
if "aircast_pdf" in st.session_state:
    st.download_button("Download PDF report",
                       data=st.session_state["aircast_pdf"],
                       file_name="aircast_report.pdf",
                       mime="application/pdf",
                       type="primary")