# Climbing Performance Evaluation & Prediction Application
This repository contains a comprehensive application for evaluating climbers' physical performance and predicting their IRCRA (International Rock Climbing Research Association) levels. The toolkit integrates force sensor and NIRS (Near-Infrared Spectroscopy) data acquisition, test evaluation, and machine learning-based performance prediction.
## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Database Structure](#database-structure)
- [Machine Learning Models](#machine-learning-models)
- [Contributing](#contributing)
- [License](#license)

## Overview
This application is developed as part of a research project to assess climbers' physical capabilities and predict their climbing performance potential. The system provides a comprehensive suite of tools for researchers to:
- Manage climber profiles and demographic information
- Conduct standardized force and physiological tests
- Visualize and analyze test results in real-time
- Generate detailed reports and export data
- Train and utilize machine learning models for performance prediction

## Features
### User Management
- Multi-level user authentication system
- Role-based access control (admin vs. researcher)
- User profile management

### Climber Management
- Registration and management of climber profiles
- Storage of demographic, experience, and physical data
- Tracking of multiple climbers per researcher

### Test Administration
- Support for various test protocols (MVC, endurance, etc.)
- Real-time data acquisition from force sensors and NIRS devices
- Test session management and organization

### Data Analysis
- Force curve analysis and metric extraction
- NIRS data processing for muscle oxygenation assessment
- Computation of key performance indicators

### Results Visualization
- Interactive data visualization of test results
- Comparison tools for performance tracking
- Detailed repetition-level analysis

### Performance Prediction
- IRCRA grade prediction using machine learning models
- Feature extraction and dimensionality reduction
- Model training interface for researchers

### Reporting
- PDF report generation for test sessions
- Data export in multiple formats (CSV, XLSX, HDF5)
- Statistical analysis and aggregation

## System Architecture
The application is structured in a modular fashion with several key components:
- **Authentication Module**: Manages user accounts and session control
- **Climber Management Module**: Handles climber profiles and information
- **Test Administration Module**: Controls test configuration and execution
- **Data Acquisition Module**: Manages sensor data collection and processing
- **Results Module**: Provides analysis and visualization of test data
- **Statistics Module**: Implements machine learning models and predictions
- **Database Layer**: Maintains structured storage of all application data

## Installation
### Prerequisites
- Python 3.11 or higher
- SQLite database engine
- Required Python packages (listed in requirements.txt)

### Setup
1. Clone the repository:
``` bash
git clone https://github.com/yourusername/climbing-performance-toolkit.git
cd climbing-performance-toolkit
```
1. Create and activate a virtual environment:
``` bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
1. Install dependencies:
``` bash
pip install -r requirements.txt
```
1. Initialize the database:
``` bash
python main.py --init-db
```
## Usage
### Running the Application
Launch the application with:
``` bash
python main.py
```

Then log in with your administrator credentials.

### Workflow
1. **Register Climbers**: Add participant information through the Research Members interface
2. **Conduct Tests**: Use the Test Page to configure and execute climbing tests
3. **View Results**: Access the Results Page to analyze test data and visualize performance
4. **Generate Reports**: Export data or create PDF reports from the Results interface
5. **Train Models**: Use the Statistics Page to train and evaluate prediction models
6. **Predict Performance**: Apply trained models to predict climber IRCRA grades

## Database Structure
The application uses SQLite databases organized into three main components:
1. **Login Database**: Stores user authentication information and research profiles
2. **Climber Database**: Contains climber demographic and experience information
3. **Tests Database**: Stores test configurations, results, and computed metrics

## Machine Learning Models
The application employs several machine learning approaches for performance prediction:
- : For dimensionality reduction of high-dimensional test data **Principal Component Analysis (PCA)**
- : For IRCRA grade prediction based on test performance **Support Vector Regression (SVR)**
- **Linear Regression**: For simplified performance modeling and comparison

Models are trained on existing data and can be continuously improved as more test data is collected.
## Contributing
Contributions to this project are welcome. Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.
_Note: This application is designed for research purposes. Always follow ethical guidelines when collecting and analyzing human subject data._
