# Trained and Analyzed an AI Model to Predict Customer Satisfaction

This project implements a machine learning solution for predicting customer satisfaction in the airline industry using various algorithms including Decision Trees, Neural Networks, Random Forest, and Clustering approaches.

## Project Overview

The project consists of two main components:
1. Data Analysis (`analysis.py`): Performs exploratory data analysis on the customer satisfaction dataset
2. Model Training (`training_prediction.py`): Implements and compares different ML algorithms for prediction

### Features Analyzed
- Customer Demographics (Age, Gender)
- Travel Details (Flight Distance, Type of Travel)
- Service Ratings (Food, WiFi, Booking Experience, etc.)
- Flight Details (Delays, Class)

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
tqdm
```

## Project Structure

```
ML-airline-satisfaction-analysis/
├── Analysis/
│   ├── Analysis.pdf
│   └── analysis.py
├── Model Training/
│   ├── Model Training.pdf
│   └── training_prediction.py
├── README.md
├── requirements.txt
└── training-data.csv
```

**Important Note about File Paths**: 
The project expects the `training-data.csv` file to be in the root directory of the project. If you encounter any path-related issues:

1. Make sure the CSV file is in the correct location (in the ML root folder)
2. Update the file paths in both Python scripts if needed:
   - In `analysis.py`: Update the path in `pd.read_csv(r"training-data.csv")`
   - In `training_prediction.py`: Update the path in `pd.read_csv("training-data.csv")`

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/NoamGanim1/ML-airline-satisfaction-analysis.git
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the analysis script:
   ```bash
   python analysis.py
   ```

4. Run the model training script:
   ```bash
   python training_prediction.py
   ```

## Analysis Components

### Data Analysis (`analysis.py`)
- Exploratory Data Analysis (EDA)
- Feature distribution analysis
- Correlation analysis
- Data visualization including histograms and correlation matrices
- Data preprocessing and cleaning

### Model Training (`training_prediction.py`)
The project implements multiple ML algorithms:

1. **Decision Tree**
   - Pre-pruning with criterion and class weight parameters
   - Hyperparameter tuning using GridSearchCV
   - Feature importance analysis

2. **Neural Networks**
   - MLPClassifier implementation
   - Hyperparameter tuning using RandomizedSearchCV
   - Learning rate and batch size optimization

3. **Random Forest**
   - GridSearchCV for parameter optimization
   - ROC curve analysis
   - Learning curve visualization

4. **Clustering**
   - K-means implementation
   - Silhouette score analysis
   - PCA for dimensionality reduction

## Model Performance

The models are evaluated using various metrics:
- F1 Score
- Confusion Matrix
- ROC Curve
- Learning Curves
- Silhouette Score (for clustering)

## Data Preprocessing

The project includes several preprocessing steps:
- Handling missing values
- Feature scaling (Standard and MinMax)
- Categorical encoding
- Feature engineering
- Outlier detection and handling

## Usage Example

```python
# Load and preprocess data
df = pd.read_csv("training-data.csv")

# Run analysis
# See analysis.py for detailed analysis steps

# Train models
# See training_prediction.py for model implementation details
```

## Authors

- Noam Ganim
