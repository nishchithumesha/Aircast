# src/satellite.py
import io, math, itertools
from typing import Tuple
import requests
from PIL import Image

# ESRI World Imagery tiles (no API key required)
ESRI_TILE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

def lonlat_to_tile(lon: float, lat: float, z: int) -> Tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2.0 ** z
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile

def fetch_tile_esri(z: int, x: int, y: int, timeout=10) -> Image.Image:
    url = ESRI_TILE.format(z=z, x=x, y=y)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def stitched_satellite(lon: float, lat: float, z: int = 12, grid: int = 3, size: int = 256) -> Image.Image:
    """
    Returns a stitched satellite image centered on (lon,lat).
    grid=3 -> 3x3 tiles; z=12 is a good starting zoom.
    """
    grid = max(1, grid)
    x0, y0 = lonlat_to_tile(lon, lat, z)
    half = grid // 2
    canvas = Image.new("RGB", (size * grid, size * grid))
    for dy, dx in itertools.product(range(-half, half + 1), range(-half, half + 1)):
        x, y = x0 + dx, y0 + dy
        try:
            tile = fetch_tile_esri(z, x, y)
            tile = tile.resize((size, size), Image.BILINEAR)
            cx = (dx + half) * size
            cy = (dy + half) * size
            canvas.paste(tile, (cx, cy))
        except Exception:
            # paste a gray placeholder on failures
            ph = Image.new("RGB", (size, size), (200, 200, 200))
            cx = (dx + half) * size
            cy = (dy + half) * size
            canvas.paste(ph, (cx, cy))
    return canvas
