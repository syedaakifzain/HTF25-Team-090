☄️ Asteroid Hazard Classification System

A data-driven machine learning system for identifying potentially hazardous asteroids using orbital and physical characteristics.









📌 Table of Contents
Overview
Problem Statement
Objectives
Solution
Key Features
Machine Learning Pipeline
Features Used
Models
Project Structure
Technology Stack
Installation
Running the Project
Streamlit Dashboard
Prediction Workflow
Model Evaluation
Expected Outputs
Dataset
Results
Team Contributions
Future Improvements
Limitations
License
Acknowledgments
🌌 Overview

The Asteroid Hazard Classification System is a machine learning project designed to classify asteroids based on whether they are potentially hazardous.

Asteroids can have very different orbital characteristics, velocities, and distances from Earth. Analysing these characteristics can help identify objects that require greater attention.

This project approaches the problem as a binary classification task, where a machine learning model learns patterns from historical asteroid data and predicts whether an asteroid belongs to a hazardous or non-hazardous category.

The system combines:

📊 Data preprocessing
🧹 Data cleaning
🔍 Exploratory data analysis
🧠 Feature selection
🤖 Machine learning
📈 Model evaluation
📉 Data visualization
🌐 Interactive Streamlit dashboard
🔮 Real-time prediction using user-provided asteroid characteristics

🎯 Problem Statement
Data-Driven Classification of Hazardous Asteroids

The objective is to build a binary classification machine learning model capable of predicting whether an asteroid is potentially hazardous using measurable orbital and physical characteristics.

The model considers attributes such as:

Relative velocity
Miss distance
Orbital period
Semi-major axis
Aphelion distance
Mean motion
Different representations of miss distance

The classification system can help demonstrate how machine learning can be applied to astronomical datasets for automated risk categorization.

💡 Objectives

The primary objectives of this project are:

1. Data Preparation

Clean and preprocess the asteroid dataset so that it can be effectively used for machine learning.

2. Feature Selection

Identify relevant orbital and physical characteristics that contribute to asteroid hazard classification.

3. Model Development

Train multiple classification algorithms and compare their performance.

4. Model Evaluation

Evaluate models using standard classification metrics such as:

Accuracy
Precision
Recall
F1-score
Confusion matrix
5. Interactive Prediction

Provide a user-friendly interface where users can enter asteroid characteristics and receive a hazard prediction.

6. Data Visualization

Present dataset statistics, distributions, correlations, and model performance through visualizations.

🧠 Solution

The system follows a complete machine learning workflow:

                    ┌──────────────────────┐
                    │    Asteroid Dataset  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Preprocessing   │
                    │ & Cleaning           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Selection    │
                    │ & Engineering        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Train/Test Split     │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
        ┌────────────┐ ┌────────────┐ ┌────────────┐
        │ Logistic   │ │ Decision   │ │ Random     │
        │ Regression │ │ Tree       │ │ Forest     │
        └──────┬─────┘ └──────┬─────┘ └──────┬─────┘
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                    ┌──────────────────────┐
                    │ Model Evaluation     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Best Model Selection │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Hazard Prediction    │
                    └──────────────────────┘
✨ Key Features
📊 1. Data Exploration

The application provides tools for exploring the asteroid dataset, including:

Dataset preview
Dataset statistics
Missing-value analysis
Feature distributions
Correlation analysis
Data summaries
🧹 2. Data Preprocessing

The preprocessing pipeline prepares raw asteroid data for machine learning.

Typical preprocessing operations include:

Selecting relevant columns
Handling missing values
Converting numerical fields
Cleaning inconsistent data
Preparing target labels
Separating features and target variables
🎯 3. Feature Selection

The model uses orbital and distance-related characteristics to identify patterns associated with hazardous asteroids.

The project README identifies the following major feature groups:

Relative velocity
Miss distance
Orbital period
Semi-major axis
Aphelion distance
Mean motion

🤖 4. Multiple Machine Learning Models

The project supports comparison between multiple classification algorithms:

Logistic Regression

Provides a simple and interpretable baseline for binary classification.

Decision Tree

Uses decision rules derived from the input features to classify asteroids.

Random Forest

Combines multiple decision trees to produce a more robust classification model.

The trained models can then be compared to determine which performs best on the available dataset.

🔬 Machine Learning Pipeline

The project follows these major stages:

Step 1 — Load Dataset

The asteroid dataset is loaded from CSV format.

dataset.csv
Step 2 — Data Cleaning

The data is inspected and cleaned before training.

Step 3 — Feature Selection

Relevant numerical features are selected.

Step 4 — Target Separation

The dataset is divided into:

X → Input features
y → Hazard classification
Step 5 — Train/Test Split

The available data is separated into training and testing subsets.

Step 6 — Model Training

Multiple classification models are trained.

Step 7 — Model Evaluation

Each model is evaluated using classification metrics.

Step 8 — Best Model Selection

The best-performing model is saved for future predictions.

Step 9 — Prediction

The selected model is used to classify new asteroid observations.

📐 Features Used

The project currently works with asteroid characteristics including:

Feature	Description
Relative Velocity km per hr	Relative velocity of the asteroid
Miles per hour	Velocity representation in miles/hour
Miss Dist. (Astronomical)	Miss distance represented in astronomical units
Miss Dist. (lunar)	Miss distance represented in lunar distances
Miss Dist. (kilometers)	Miss distance represented in kilometres
Semi Major Axis	Semi-major axis of the asteroid's orbit
Aphelion Dist	Maximum orbital distance
Mean Motion	Orbital motion parameter
Orbital Period	Time required to complete an orbit

The exact feature-selection logic is implemented inside the preprocessing pipeline.

.

🤖 Machine Learning Models

The project is designed to train and compare:

Model	Purpose
Logistic Regression	Baseline binary classifier
Decision Tree	Rule-based classification
Random Forest	Ensemble-based classification
Best Model	Selected based on evaluation performance

The resulting trained models are stored under the models/ directory.
