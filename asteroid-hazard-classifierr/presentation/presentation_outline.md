# Asteroid Hazard Classification — Presentation Outline

## Slide 1: Title
- Asteroid Hazard Classification System
- Team Name: [Your Team]
- Hackathon: CBIT Hacktoberfest 2025

## Slide 2: Problem Statement
- What are hazardous asteroids?
- Why classify them: risk assessment, mission planning, early warning systems
- Objective: Predict Hazardous (True/False) from orbital/physical features

## Slide 3: Dataset Overview
- 4,534 asteroids
- 24 features (velocity, distances, orbital parameters)
- Target: `Hazardous` (binary)
- Source format: CSV

## Slide 4: Our Approach
- Data Pipeline: Load → Clean → Train → Evaluate
- Tools: Python, pandas, numpy, scikit-learn, matplotlib, seaborn
- Reproducible pipelines and saved artifacts

## Slide 5: Data Preprocessing
- Missing value handling: numeric→median, categorical→mode
- Categorical encoding: LabelEncoder for selected columns
- Feature selection: key orbital and distance features
- Scaling: StandardScaler

## Slide 6: Models Trained
- Logistic Regression (max_iter=1000)
- Decision Tree (max_depth=10)
- Random Forest (n_estimators=100, max_depth=15, min_samples_split=5)

## Slide 7: Evaluation & Results
- Metrics: Accuracy, Precision, Recall, F1
- Confusion matrix visualization
- Best model selection by accuracy
- Accuracy achieved: TBD%

## Slide 8: Team Contributions
- Data: Cleaning & preprocessing
- Modeling: Training & tuning
- Analysis: EDA & insights
- Presentation: Slides & demo

## Slide 9: Demo & Files
- Run: `python main.py`
- Outputs: models, confusion matrix PNG, classification report TXT
- Notebook: `notebooks/exploratory_analysis.ipynb`

## Slide 10: Acknowledgments & License
- Thanks: CBIT Hacktoberfest 2025
- License: MIT