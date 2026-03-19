# March Madness Learning

**CSC 480 Final Project — Cal Poly San Luis Obispo**  
**Instructor:** Dr. Rodrigo Canaan  
**Team Members:** Nathan Huang, Casey Hartley, Vardaan Kumar

## Project Overview

This repository contains our CSC 480 final project on predicting NCAA March Madness tournament outcomes using machine learning. The main focus of the project was to compete in the Kaggle **March Machine Learning Mania 2026** competition by building models that predict matchup win probabilities from historical regular-season and tournament data.

Our system builds season-level team features from historical box score statistics, tournament seeds, and Massey rankings, then converts tournament matchups into training examples. We explored three separate modeling approaches: **XGBoost, Random Forest, and MLP**. After generating win probabilities for individual matchups, we extended the project by using those predictions to produce a full tournament bracket.

## Acknowledgments and External Resources

This project was completed as part of **CSC 480 at Cal Poly**.

This project uses data from the Kaggle **March Machine Learning Mania 2026** competition.

Jeff Sonas, Martyna Plomecka, Yao Yan, and Addison Howard. *March Machine Learning Mania 2026*. https://kaggle.com/competitions/march-machine-learning-mania-2026, 2026. Kaggle.

The competition datasets are not included in this repository. Users should download the data directly from the official Kaggle competition page after agreeing to the competition rules.

This repository includes only code and documentation. No Kaggle competition data is redistributed here.

### External Resources

We built on the following external resources:
- **XGBoost** library and the XGBoost paper
- Python libraries for machine learning and data processing, including:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `matplotlib`
  - `scipy`
  - `joblib`
  - `torch`

We also consulted prior March Madness prediction work, including:

1. Chen, T. (2016). XGBoost: A Scalable Tree Boosting System.
2. Gumm, J., Barrett, A., & Hu, G. (2015). A machine learning strategy for predicting march madness winners.
3. Shen, G., Gao, D., Wen, Q., & Magel, R. (2016). Predicting Results of March Madness Using Three Different Methods.
4. Kim, J. W., Magnusen, M., & Jeong, S. (2023). March Madness prediction: Different machine learning approaches with non-box score statistics.

## Repository Structure

### Build Training Table
- `features.py` — builds team-level and matchup-level features from raw March Madness data

### XGBoost files
- `train_xgb_win.py` — trains the XGBoost model for win probability prediction
- `train_xgb_margin.py` — trains the XGBoost model for margin-related prediction
- `hpt_win.py` — hyperparameter tuning for the win model
- `hpt_margin.py` — hyperparameter tuning for the margin model
- `submission.py` — generates Kaggle-style prediction outputs
- `bracket.py` — uses predicted matchup probabilities to generate a tournament bracket

### Random Forest file
- `random_forest.py` — Random Forest model implementation for tournament prediction

### MLP file
- `MMFinal.py` — MLP model implementation for tournament prediction

## Data Setup

The large Kaggle dataset files are **not included in this repository**. To run the project, download the March Machine Learning Mania data separately and place the CSV files in a local folder such as:

```bash
mmlm2026/
```

The project may also create intermediate or output files in a folder such as:
```bash
data/
```

## Installation

```bash
git clone https://github.com/nthuang/march-madness-predictor.git
cd march-madness-predictor
pip install numpy pandas scikit-learn xgboost matplotlib scipy joblib torch
```
## How to Run the Project

The project includes three separate modeling approaches, so the workflow depends on which model you want to run.

### Build features
Generate the season-level team features and matchup-level training data used by all models:

1. Download the Kaggle competition data
2. Place the files in `mmlm2026/`
3. Run:
```bash
python features.py
```

### Run the XGBoost pipeline
Optional hyperparameter tuning:
```bash
python hpt_win.py
python hpt_margin.py
```

Optional standalone training and evaluation:
```bash
python train_xgb_win.py
python train_xgb_margin.py
```

Generate final matchup predictions:
```bash
python submission.py
```

Generate tournament bracket:
```bash
python bracket.py
```

### Run Random Forest Mode
```bash
python random_forest.py
```

### MLP Model
```bash
python MMFinal.py
```
