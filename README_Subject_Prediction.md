# Subject Prediction from Academic Abstracts

This project implements a multi-label classification system to predict academic subjects from paper abstracts. It handles both English and German text using transformer-based models.

## Project Structure

subject-prediction/
├── data/
│ ├── raw/ # Original JSON data
│ └── processed/ # Preprocessed data files
├── src/
│ ├── data_preparation/
│ │ └── preprocessing.py
│ ├── modeling/
│ │ └── train_model.py
│ └── evaluation/
│ └── evaluate_model.py
├── notebooks/ # Jupyter notebooks for analysis
├── tests/ # Unit tests
├── requirements.txt # Project dependencies
├── README_Subject_Prediction.md # Project documentation
└── .gitignore # Files to ignore in the repo

## Setup and Installation

1. Create and activate a virtual environment:

```bash
python3 -m venv env
source env/bin/activate
