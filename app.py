import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎", layout="centered")


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("models/preprocessor.pkl")
    model = joblib.load("models/rf_model.pkl")
    return preprocessor, model


try:
    preprocessor, model = load_artifacts()
except Exception as e:
    st.error(f"Could not load the model files. Details: {e}")
    st.stop()

st.title("💎 Diamond Price Predictor")

with st.expander("How to use this tool", expanded=True):
    st.markdown(
        """
        1. Enter the diamond's details in the form below.
        2. Click **Predict Price**.
        3. The estimated price (in US dollars) appears instantly.

        The model is a Random Forest trained on about 54,000 diamonds.
        """
    )

with st.form("diamond_form"):
    col1, col2 = st.columns(2)

    with col1:
        carat = st.number_input("Carat (weight)", min_value=0.0, max_value=6.0, value=0.7, step=0.01)
        cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=4)
        color = st.selectbox("Color (D = best, J = worst)", ["D", "E", "F", "G", "H", "I", "J"], index=3)
        clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], index=2)

    with col2:
        depth = st.number_input("Depth %", min_value=0.0, max_value=100.0, value=61.8, step=0.1)
        table = st.number_input("Table %", min_value=0.0, max_value=100.0, value=57.0, step=0.1)
        x = st.number_input("Length x (mm)", min_value=0.0, max_value=15.0, value=5.7, step=0.01)
        y = st.number_input("Width y (mm)", min_value=0.0, max_value=15.0, value=5.7, step=0.01)
        z = st.number_input("Height z (mm)", min_value=0.0, max_value=10.0, value=3.5, step=0.01)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    # ---- Input validation ----
    errors = []
    if carat <= 0:
        errors.append("Carat must be greater than 0.")
    if x <= 0 or y <= 0 or z <= 0:
        errors.append("Length, width and height must all be greater than 0.")
    if not (40 <= depth <= 80):
        errors.append("Depth % should be between 40 and 80.")
    if not (40 <= table <= 100):
        errors.append("Table % should be between 40 and 100.")

    if errors:
        for msg in errors:
            st.error(msg)
    else:
        try:
            input_df = pd.DataFrame(
                [{
                    "carat": carat, "cut": cut, "color": color, "clarity": clarity,
                    "depth": depth, "table": table, "x": x, "y": y, "z": z,
                }]
            )
            processed = preprocessor.transform(input_df)
            prediction = float(model.predict(processed)[0])

            st.success("Prediction complete")
            st.metric("Estimated Price", f"${prediction:,.2f}")

            # Show the range of predictions across the individual trees in the forest
            tree_preds = [tree.predict(processed)[0] for tree in model.estimators_]
            low, high = min(tree_preds), max(tree_preds)
            st.caption(f"Individual trees predicted between ${low:,.0f} and ${high:,.0f}.")

            # Soft warning if the measurements look physically inconsistent
            expected_depth = 2 * z / (x + y) * 100
            if abs(expected_depth - depth) > 5:
                st.warning(
                    "Your measurements and depth % don't match well "
                    f"(expected about {expected_depth:.1f}%). Double-check your inputs."
                )
        except Exception as e:
            st.error(f"Something went wrong while predicting: {e}")
