# AirCast

AirCast is a Streamlit dashboard for exploring hourly air-quality data alongside satellite imagery. Enter a location, fetch recent pollutant measurements, inspect trends and summary statistics, generate a simple 24-hour forecast, and export the results as CSV or PDF.

## Features

- Fetches the previous 1-7 days of hourly pollutant data from the [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api).
- Supports PM2.5, PM10, nitrogen dioxide, ozone, sulphur dioxide, and carbon monoxide.
- Displays a latest-value status indicator using Good, Moderate, Bad, and Critical ranges.
- Builds a satellite mosaic centered on the selected coordinates.
- Uses EOX satellite tiles first, with NASA GIBS and Esri World Imagery fallbacks.
- Shows interactive Plotly time-series charts and summary statistics.
- Produces a basic per-pollutant 24-hour linear-regression forecast.
- Downloads raw hourly data and summary statistics as CSV files.
- Generates a PDF report containing metadata, satellite imagery, summary statistics, forecast tables, and sample raw rows.
- Includes a Gemini-powered assistant that can answer questions using recent air-quality data.

## Requirements

- Python 3.10 or newer
- Internet access for Open-Meteo, satellite tiles, and Gemini requests
- A Google Gemini API key for the assistant feature

## Installation

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/nishchithumesha/Aircast.git
cd Aircast
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

The chatbot requires a Gemini API key. Do not commit API keys to source control. The current version of the app contains a key in the source code; revoke and rotate that key in Google AI Studio before sharing or deploying this repository, then update the application to read the replacement from an environment variable or Streamlit secret.

The air-quality and satellite providers used by AirCast do not require an application key, but their requests still require network access and may be affected by rate limits or firewall rules.

## Run the application

From the repository directory, run:

```powershell
streamlit run main.py
```

Streamlit will print a local URL, normally `http://localhost:8501`.

## Using the dashboard

1. Enter a latitude and longitude. The default location is Bengaluru, India.
2. Select the number of historical days and the pollutants to retrieve.
3. Adjust satellite zoom, mosaic grid size, and tile size if needed.
4. Select **Run** to fetch data and render the dashboard.
5. Use the sidebar assistant to ask questions about the loaded data.
6. Download raw data, summary statistics, or the generated PDF report.

Use **Test satellite connectivity** to check whether the satellite tile providers are reachable from your network.

## Data and model notes

- Open-Meteo supplies modelled hourly air-quality data. AirCast does not replace official monitoring stations or medical advice.
- The status ranges are applied to the latest selected pollutant value using the app's configured thresholds: 0-50 Good, 51-100 Moderate, 101-200 Bad, and above 200 Critical.
- The forecast is intentionally simple: it fits an independent linear regression to each pollutant's historical values and projects 24 hours forward. It should be treated as an exploratory trend estimate, not a validated prediction.
- Satellite imagery is fetched as map tiles and stitched into a mosaic. Provider availability and image coverage can vary.

## Project layout

```text
main.py            Streamlit application entry point
src/               Supporting Open-Meteo, satellite, and utility modules
requirements.txt   Python dependencies
.gitignore         Local environments, caches, and secrets excluded from Git
```

## License

No license has been specified for this repository yet. Add a license file before distributing the project or accepting external contributions.