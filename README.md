# 💎 Diamond Price Predictor

An interactive web app that predicts the price of a diamond in real time from
its physical features (carat, cut, color, clarity, depth, table and dimensions).
Built for the AI/ML Technical Team task round.

**Live demo:** <your Streamlit app link>
**Training notebook (Task 1):** <your Task 1 GitHub repo link>

## How to Use
1. Enter the diamond's details in the form.
2. Click **Predict Price**.
3. The estimated price (in US dollars) is shown instantly, along with the
   range predicted by the individual trees in the model.

Inputs are validated, and clear error messages appear for invalid values
such as zero or negative sizes.

## Dataset
Seaborn's built-in `diamonds` dataset (about 54,000 rows, 10 columns).
Target variable: `price`.
Cleaning included removing duplicates, invalid zero dimensions and
impossible outliers in the y and z columns.

## Model
- Algorithm: Random Forest Regressor (compared against Linear Regression
  and Decision Tree in the training notebook)
- Preprocessing: StandardScaler for numeric features and OneHotEncoder for
  categorical features, both fitted on the training data only to avoid
  data leakage
- Saved with `joblib` (`models/preprocessor.pkl`, `models/rf_model.pkl`)

### Performance (test set)
| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | <value> | <value> | <value> |
| Random Forest | <value> | <value> | <value> |

Random Forest performed best because it captures the non-linear
relationship between carat and price.

## Tech Stack
Python, Pandas, NumPy, Scikit-Learn, Joblib, Streamlit

Dependencies are listed in `requirements.txt`.

## Run Locally
1. Clone the repo:
   `git clone <your repo link>`
   `cd diamond-price-app`
2. (Optional) create a virtual environment:
   `python -m venv venv`, then activate it
3. Install dependencies:
   `pip install -r requirements.txt`
4. Start the app:
   `streamlit run app.py`
5. Open the local URL shown in the terminal (usually http://localhost:8501).

## Reproducing the Model
The full data cleaning, EDA and training pipeline is in the Task 1 notebook:
<your Task 1 repo link>. Running it end to end regenerates the files in `models/`.

## Project Structure
```
diamond-price-app/
├── app.py
├── requirements.txt
├──
