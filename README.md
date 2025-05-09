# 🧗‍♀️ Climbing Performance Evaluation & Prediction Toolkit

This repository contains the source code and documentation for a comprehensive system designed to evaluate climbers' physical performance and predict their IRCRA levels. The toolkit integrates force and NIRS data acquisition, test evaluation, and machine learning-based performance prediction.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data Import](#-data-import)
- [Model Training](#-model-training)
- [Prediction](#-prediction)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 Overview

The system is developed as part of a master's thesis project, focusing on the assessment of climbers' physical capabilities and the prediction of their climbing performance levels. It provides tools for:

- Importing and managing climber data
- Processing force and NIRS test data
- Evaluating test results to extract key metrics
- Training machine learning models to predict IRCRA levels
- Generating comprehensive reports

---

## 🚀 Features

- **Data Management**: Structured storage of climber profiles and test results using SQLite databases.
- **Data Import**: Scripts to import data from Excel files and existing databases.
- **Test Evaluation**: Automated computation of metrics like MVC, RFD, and critical force.
- **Prediction Models**: Training and evaluation of Linear Regression and SVR models for IRCRA level prediction.
- **Visualization**: Graphical representation of test data and prediction results.

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/ancasebe/thesis.git
cd thesis
```
Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🧪 Usage

The application is modular, with separate scripts for different functionalities:

Data Import: Use exel_db_manager.py to import data from Excel files.
Test Evaluation: Run evaluation scripts to process test data and compute metrics.
Model Training: Train prediction models using the provided training scripts.
Prediction: Use trained models to predict IRCRA levels for new test data.

---

## 📥 Data Import

To import data from an Excel file:

python excel_db_manager.py path_to_excel_file.xlsx
This script will:

Read test data and participant metadata from the Excel file
Create entries in the climber and test databases
Compute evaluation metrics and store them in the database

---

## 🧠 Model Training

To train the prediction models:

Ensure that the databases are populated with evaluated test data
Run the training script:
```bash
python ircra_prediction_model.py
```
This will:

Extract features and labels from the database
Perform PCA for dimensionality reduction
Train Linear Regression and SVR models
Evaluate model performance and save the trained models

---

## 📈 Prediction

To predict the IRCRA level for a new test:

Ensure the test data is evaluated and stored in the database
Run the prediction script:
python predict_irca.py test_id
This will:

Load the trained model
Extract features for the specified test
Predict the IRCRA level and display the result

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository and submit a pull request.

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
