# 🏡 Ames Housing Price Prediction
//SAMBH
This project uses the **Ames Housing dataset** to predict house prices based on a wide range of features.  
It follows the full ML pipeline: **EDA → Feature Engineering → Feature Selection → Baseline Modeling → Future Extensions**.

---

## 📂 Repository Structure

```
data/
    raw/                # Original Kaggle data (not committed to Git)
    processed/          # Cleaned and engineered datasets
notebooks/
    01_eda_on_train.ipynb
    02_feature_eng_on_train.ipynb
    03_eda_feature_eng_on_test.ipynb
    04_feature_selection.ipynb
    05_train_test_compare_baseline.ipynb
artifacts/              # Encoders, scalers, mappings

## ✅ Steps Completed
1. **Exploratory Data Analysis (EDA)**
   - Inspected distributions, correlations, outliers, missing values.
2. **Feature Engineering**
   - Numeric imputation (median).
   - Categorical imputation (`None`, mode, or "Rare_var").
   - Encoding (LabelEncoder with unseen mapped to -1).
   - Scaling (MinMaxScaler).
3. **Feature Selection**
   - Reduced to ~39 most informative features.
4. **Baseline Model**
   - Trained Linear Regression baseline.
   - Achieved RMSE ≈ 0.14 (log-transformed target).

---

## 🚀 Next Steps
- Train advanced models: RandomForest, XGBoost, LightGBM, CatBoost.
- Perform hyperparameter tuning (GridSearchCV, Optuna).
- Consider log-transforming `SalePrice` for better RMSE.
- Explore feature interactions and domain-driven features.
- Wrap preprocessing into a reproducible `Pipeline`.
- Perform residual analysis to spot systematic errors.
- Finalize model and generate submission file with predictions.

---

## ⚙️ Requirements
Install dependencies via:
```bash
pip install -r requirements.txt
```

Main packages:
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- xgboost, lightgbm, catboost (for future steps)
- joblib

---

## 📊 Results
- **Baseline Linear Regression**: RMSE ≈ 0.143, MAE ≈ 0.099 (log scale).
- Advanced models (to be added) expected to improve results.


## 🙌 Acknowledgments
- Dataset: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
