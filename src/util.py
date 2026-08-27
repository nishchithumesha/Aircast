# src/util.py
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["variable","count","mean","median","min","max","std"])
    g = df.groupby("variable")["value"]
    out = pd.DataFrame({
        "count": g.count(),
        "mean": g.mean(),
        "median": g.median(),
        "min": g.min(),
        "max": g.max(),
        "std": g.std(),
    }).reset_index()
    return out

def export_pdf(df: pd.DataFrame, summary: pd.DataFrame, path: str, title: str, meta: str, sat_png_path: str | None = None):
    c = canvas.Canvas(path, pagesize=A4)
    W, H = A4
    margin = 1.5*cm
    y = H - margin

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, title)
    y -= 0.8*cm
    c.setFont("Helvetica", 9)
    c.drawString(margin, y, meta)
    y -= 0.8*cm

    # Satellite image block (optional)
    if sat_png_path:
        try:
            img = ImageReader(sat_png_path)
            # Fit within page width, keep aspect
            max_w = W - 2*margin
            max_h = 7*cm
            c.drawImage(img, margin, y - max_h, width=max_w, height=max_h, preserveAspectRatio=True, anchor='n')
            y -= (max_h + 0.5*cm)
        except Exception:
            c.setFont("Helvetica-Oblique", 9)
            c.drawString(margin, y, "(Satellite image failed to render)")
            y -= 0.6*cm

    # Summary table
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Summary by variable"); y -= 0.6*cm
    c.setFont("Helvetica-Bold", 9)
    headers = ["variable","count","mean","median","min","max","std"]
    x_positions = [margin, margin+4*cm, margin+6*cm, margin+8*cm, margin+10*cm, margin+12*cm, margin+14*cm]
    for h, x in zip(headers, x_positions):
        c.drawString(x, y, h.upper())
    y -= 0.5*cm
    c.setFont("Helvetica", 9)
    for _, r in summary.iterrows():
        vals = [str(r[h]) if h=="variable" else f"{r[h]:.2f}" for h in headers]
        for val, x in zip(vals, x_positions):
            c.drawString(x, y, val)
        y -= 0.4*cm
        if y < 2*cm:
            c.showPage(); y = H - margin
            c.setFont("Helvetica-Bold", 11); c.drawString(margin, y, "Summary (cont.)"); y -= 0.6*cm
            c.setFont("Helvetica", 9)

    # New page for sample rows
    c.showPage()
    c.setFont("Helvetica-Bold", 11); y = H - margin
    c.drawString(margin, y, "Sample (first 40 rows)"); y -= 0.6*cm
    c.setFont("Helvetica", 8)
    for _, r in df.head(40).iterrows():
        unit = r.get("unit","") if pd.notna(r.get("unit","")) else ""
        line = f"{r['time']} | {r['variable']} = {r['value']} {unit}"
        c.drawString(margin, y, line)
        y -= 0.35*cm
        if y < 2*cm:
            c.showPage(); y = H - margin
            c.setFont("Helvetica", 8)

    c.save()
