<<<<<<< HEAD
# Asteroid Hazard Classification System ☄️

Predict hazardous asteroids using orbital and physical characteristics.

## Problem Statement (PS26)
Data-Driven Classification of Hazardous Asteroids (Beginner Level). Build a binary classification ML model that predicts whether an asteroid is hazardous based on its velocity, distance, and orbital parameters.

## Team (CBIT Hacktoberfest 2025)
- Member 1 — Role: Data Engineer
- Member 2 — Role: ML Engineer
- Member 3 — Role: Analyst & Visualization
- Member 4 — Role: Presenter & Documentation

## Technologies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Folder Structure
```
asteroid-hazard-classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── data/
│   └── dataset.csv (place file here)
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── model_evaluation.py
├── models/
│   └── (trained models saved here)
├── results/
│   ├── confusion_matrix.png
│   └── classification_report.txt
├── notebooks/
│   └── exploratory_analysis.ipynb
└── presentation/
    └── presentation_outline.md
```

## Installation
- Ensure Python 3.8+ is installed.
- Install dependencies:
```
pip install -r requirements.txt
```

## How to Run
```
python main.py
```

## Expected Outputs
- Trained model files in `models/` (Logistic Regression, Decision Tree, Random Forest, and `best_model.pkl`)
- `results/confusion_matrix.png` (visual confusion matrix)
- `results/classification_report.txt` (precision/recall/F1 report)

## Key Features Used
- Relative Velocity
- Miss Distance
- Orbital Period
- Semi Major Axis
- Aphelion Distance

(Specific column names selected in preprocessing: `Relative Velocity km per hr`, `Miles per hour`, `Miss Dist.(Astronomical)`, `Miss Dist.(lunar)`, `Miss Dist.(kilometers)`, `Semi Major Axis`, `Aphelion Dist`, `Mean Motion`, `Orbital Period`)

## Results
- Accuracy: TBD% (after training on the provided dataset)

## Team Contributions
- Data ingestion and cleaning
- Feature selection and engineering
- Model training and evaluation
- Visualization and documentation

## License
MIT License

## Acknowledgment
Special thanks to CBIT Hacktoberfest 2025 for the dataset context and hackathon motivation.

---

## Streamlit Web App
A professional, demo-ready dashboard is available.

### Run Locally
```
pip install -r requirements.txt
streamlit run app.py --server.headless true --server.port 8501
```

### Features
- Home overview with metrics and model accuracy chart (Plotly)
- Data exploration: preview, describe(), missing values heatmap, distributions, correlations
- Models: run training pipeline, live logs, comparisons, confusion matrix, downloads
- Predict: interactive sliders for all features, colored results, confidence & contributions
- About: team, tech, links

### Offline Ready
- Loads `data/dataset.csv` (or root `dataset.csv`)
- Uses local models in `models/` and results in `results/`
- No external APIs required
=======
# 90_lapsus
>>>>>>> ab57a2ff56068248e4b0e4a5505252d552a49a0c
