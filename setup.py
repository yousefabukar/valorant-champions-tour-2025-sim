import os

def create_directories():
    """Create basic project structure"""
    directories = [
        "data/raw",
        "data/processed", 
        "src",
        "web/static",
        "web/templates",
        "models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/

# Data
data/raw/*.json
data/raw/*.csv
*.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    print("✅ Created: .gitignore")

def create_readme():
    """Create simple README"""
    readme_content = """# VCT Champions 2025 Predictor

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
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content.strip())
    print("✅ Created: README.md")

if __name__ == "__main__":
    print("🚀 Setting up VCT Predictor project...")
    create_directories()
    create_gitignore() 
    create_readme()
    print("\n🎉 Project setup complete!")
    print("\nNext steps:")
    print("1. pip install -r requirements.txt")
    print("2. python src/scraper.py")