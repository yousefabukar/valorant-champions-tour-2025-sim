# VCT Champions 2025 Predictor

Simple ML-powered prediction system for VALORANT Champions Tour.

## Quick Start

1. **Setup**
```bash
pip install -r requirements.txt
python setup.py
```

2. **Scrape Data**
```bash
python src/scraper.py
```

3. **Run Predictions**
```bash
python src/model.py
```

4. **Start Web App**
```bash
python web/app.py
```

## Features

- 📊 Real-time match data scraping
- 🤖 ML-powered win predictions  
- 🎮 Match-by-match simulation
- 🌐 Simple web interface

## Project Structure

```
vct-predictor/
├── data/           # Scraped and processed data
├── src/            # Core Python modules
├── web/            # Flask web app
└── models/         # Trained ML models
```