# March Madness Learning

**CSC 480 Final Project — Cal Poly San Luis Obispo**  
**Instructor:** Dr. Rodrigo Canaan  
**Team Members:** Casey Hartley, Nathan Huang, Vardaan Kumar

## Project Overview

This repository contains our CSC 480 project implementation for predicting NCAA March Madness tournament outcomes using machine learning. The project was developed around the Kaggle **March Machine Learning Mania** competition and focuses on predicting matchup win probabilities from historical regular-season and tournament data.

Our final system builds season-level team features from historical box score statistics, tournament seeds, and Massey rankings, then converts tournament matchups into training instances. We compared multiple models, including Random Forest, XGBoost, and MLP, and found that XGBoost performed best for our final setup.

## Acknowledgments and External Resources

This project was completed as part of **CSC 480 at Cal Poly**.

We built on the following external resources:

- **Kaggle March Machine Learning Mania** competition and datasets
- **XGBoost** library and the XGBoost paper
- Python libraries for machine learning and data processing, including:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `matplotlib` (if used for analysis/plots)

We also consulted prior March Madness prediction work, including:

1. Chen, T. (2016). XGBoost: A Scalable Tree Boosting System.
2. Gumm, J., Barrett, A., & Hu, G. (2015). A machine learning strategy for predicting march madness winners.
3. Shen, G., Gao, D., Wen, Q., & Magel, R. (2016). Predicting Results of March Madness Using Three Different Methods.
4. Kim, J. W., Magnusen, M., & Jeong, S. (2023). March Madness prediction: Different machine learning approaches with non-box score statistics.

## Repository Structure

Main scripts in this repository:

- `features.py` — builds team-level and matchup-level features from raw March Madness data
- `train_xgb_win.py` — trains the XGBoost model for win probability prediction
- `train_xgb_margin.py` — trains the XGBoost model for margin-related prediction
- `hpt_win.py` — hyperparameter tuning for the win model
- `hpt_margin.py` — hyperparameter tuning for the margin model
- `submission.py` — generates Kaggle-style prediction outputs
- `bracket.py` — uses predicted matchup probabilities to generate a tournament bracket

## Data Setup

The large Kaggle dataset files are **not included in this repository**. To run the project, download the March Machine Learning Mania data separately and place the CSV files in a local folder such as:

mmlm2026/

The project may also create intermediate or output files in a folder such as:

data/

Example input files include:

- `MNCAATourneyCompactResults.csv`
- `MRegularSeasonDetailedResults.csv`
- `MNCAATourneySeeds.csv`
- `MMasseyOrdinals.csv`

These are excluded from github because files exceed GitHub's file size limits.

## Installation

```bash
git clone https://github.com/nthuang/march-madness-predictor.git
cd march-madness-predictor


pip install numpy pandas scikit-learn xgboost matplotlib
```
## How to Run the Project

### 1. Build features
Generate season-level team features and matchup-level training data:

```bash

python features.py

#tune hyperparameters(optional), training already has best parameters set
python hpt_win.py
python hpt_margin.py

#train XGBoost models
python train_xgb_win.py
python train_xgb_margin.py

#generate predictions
python submission.py

#generate bracket
python bracket.py
```
## 4. Reproducing Results

The main results in our report were produced using season-based cross-validation, where one full season is held out at a time and the model is trained on the remaining seasons. This evaluation setup better reflects the real task of predicting a future March Madness tournament using only past data.

To reproduce the main workflow:

1. Download the Kaggle competition data
2. Place the files in `mmlm2026/`
3. Run `python features.py`
4. Optionally run hyperparameter tuning with `python hpt_win.py` and `python hpt_margin.py`
5. Train the final models with `python train_xgb_win.py` and `python train_xgb_margin.py`
6. Generate predictions with `python submission.py`
7. Generate a bracket with `python bracket.py`
