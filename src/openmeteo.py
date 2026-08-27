
import requests
import pandas as pd
from typing import List, Dict, Any, Optional

BASE = "https://air-quality-api.open-meteo.com/v1/air-quality"
DEFAULT_VARS = ["pm2_5","pm10","nitrogen_dioxide","ozone","sulphur_dioxide","carbon_monoxide"]

def fetch_hourly(lat: float, lon: float, past_days: int = 3,
                 variables: Optional[List[str]] = None,
                 timezone: str = "auto") -> Dict[str, Any]:
    vars_list = variables or DEFAULT_VARS
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(vars_list),
        "past_days": max(1, min(7, int(past_days))),
        "timezone": timezone
    }
    r = requests.get(BASE, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def to_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
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
