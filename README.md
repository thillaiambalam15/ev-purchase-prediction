# Predicting EV Purchases: Baseline vs. Boosting

Source: [Kaggle Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9) - binary classification, predicting `Will_Buy_EV` from demographic/behavioral features. Official competition metric: ROC-AUC.

## Approach

One self-contained notebook, no separate scripts. In order:

1. **EDA** on 668,665 rows / 14 columns (no missing values, ~17.5% positive class)

2. **Feature engineering**, 8 new features, each built from a stated hypothesis

3. **A 2x2 ablation**: {Logistic Regression, LightGBM} x {raw features, engineered features}, on the same stratified 80/20 split, to answer if the engineered features improve the performance

4. **Bayesian hyperparameter tuning** (Optuna, 50 trials x 5-fold CV) for both LightGBM and XGBoost.

5. **A 4-model stacking ensemble** - tuned LightGBM + tuned XGBoost + Random Forest + a small ANN (`MLPClassifier`), combined two ways: a simple probability average, and a linear (Logistic Regression) meta-model trained on out-of-fold predictions

6. **One consolidated results table + chart** covering all 12 models for comparison at a galnce

## Results

Full ranked comparison (ROC-AUC, held-out 20% test set), in sorted order:

| Model                          | ROC-AUC | PR-AUC |

| ------------------------------ | ------- | ------ |

| Stacking base: XGBoost         | 0.9417  | 0.7565 |

| XGBoost + engineered + tuned   | 0.9416  | 0.7567 |

| Stacked ensemble (4 models)    | 0.9415  | 0.7547 |

| Stacking base: LightGBM        | 0.9415  | 0.7551 |

| LightGBM + raw features        | 0.9414  | 0.7549 |

| LightGBM + engineered features | 0.9414  | 0.7544 |

| LightGBM + engineered + tuned  | 0.9414  | 0.7545 |

| Simple average (4 models)      | 0.9408  | 0.7499 |

| Stacking base: Random Forest   | 0.9390  | 0.7424 |

| Stacking base: ANN             | 0.9385  | 0.7407 |

| LogReg + engineered features   | 0.9380  | 0.7400 |

| LogReg + raw features          | 0.9380  | 0.7399 |

## Conclusion

- Boosting (LightGBM/XGBoost, 0.9414–0.9417) clearly beats Logistic Regression (0.9380), Random Forest (0.9390), and the ANN (0.9385) but boosting variants are now interchangeable with each other
- The ensemble (0.9408–0.9415) never beat the single best model (tuned XGBoost, 0.9417). Random Forest and the ANN were too weak to add useful diversity in the ensembling and they might have dragged the average down
- One lesson for myself: ensembling only pays off when our base models are comparably strong, diversity alsone from a weak model doesn't offset the accuracy it loses you

## Tools used

- **pandas / numpy** - data loading, feature engineering

- **matplotlib / seaborn** - EDA plots, ROC curves, results charts

- **scikit-learn** - `Pipeline` / `ColumnTransformer` for preprocessing, `LogisticRegression`, `RandomForestClassifier`, `MLPClassifier`, `StratifiedKFold`, evaluation metrics

- **LightGBM** and **XGBoost** - gradient boosting models

- **Optuna** - Bayesian hyperparameter search (TPE sampler) for both boosting models

## Running it

```bash

pip install -r requirements.txt

jupyter notebook notebooks/ev_purchase_boosting_vs_baseline.ipynb

```

Run all cells top to bottom. Full run time is dominated by the Optuna tuning (Sections 10-11, 50 trials x 5-fold CV each) and the stacking ensemble (Section 12, which includes a neural net) - reduce `N_TRIALS`/`N_FOLDS` in those cells, if you need a faster run.
