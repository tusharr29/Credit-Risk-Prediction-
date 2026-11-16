import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------------------------------
# Credit Risk Prediction - Streamlit demonstration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="📊",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "credit_risk_model.pkl")
    num_imputer = joblib.load(MODEL_DIR / "num_imputer.pkl")
    cat_imputer = joblib.load(MODEL_DIR / "cat_imputer.pkl")
    encoder = joblib.load(MODEL_DIR / "encoder.pkl")
    return model, num_imputer, cat_imputer, encoder

st.title("Credit Risk Prediction")
st.write(
    "An educational machine-learning demonstration that estimates whether "
    "an applicant belongs to the higher-risk class defined by this project's "
    "historical credit-status proxy."
)

st.warning(
    "This is a portfolio demonstration only. The target is an engineered "
    "proxy based on historical credit statuses, not actual loan approval "
    "outcomes. Do not use this application for real lending or financial decisions."
)

try:
    model, num_imputer, cat_imputer, encoder = load_artifacts()
except Exception as error:
    st.error(
        "Could not load the model files. Make sure the four `.pkl` files are "
        "inside the `models/` folder and that the filenames match the project instructions."
    )
    st.code(str(error))
    st.stop()

# The feature names and order must match the training notebook.
numerical_cols = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "CNT_FAM_MEMBERS",
    "AGE",
    "EMPLOYMENT_YEARS",
    "IS_EMPLOYED",
]

categorical_cols = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "AGE_GROUP",
    "INCOME_GROUP",
]

# Read category values directly from the fitted OneHotEncoder.
try:
    fitted_ohe = encoder.named_transformers_["cat"]
    category_map = {
        column: [str(value) for value in categories]
        for column, categories in zip(categorical_cols, fitted_ohe.categories_)
    }
except Exception:
    st.error("Could not read categories from the saved encoder. Check that encoder.pkl is the fitted ColumnTransformer.")
    st.stop()

with st.form("applicant_form"):
    st.subheader("Applicant information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=35)
        annual_income = st.number_input(
            "Annual income", min_value=0.0, value=150000.0, step=10000.0
        )
        children = st.number_input("Number of children", min_value=0, max_value=20, value=0)
        family_members = st.number_input(
            "Number of family members", min_value=1, max_value=20, value=2
        )
        employed = st.selectbox("Currently employed?", ["Yes", "No"])

    with col2:
        employment_years = st.number_input(
            "Years employed", min_value=0.0, max_value=60.0, value=5.0, step=0.5
        )
        gender = st.selectbox("Gender", category_map["CODE_GENDER"])
        owns_car = st.selectbox("Owns a car?", category_map["FLAG_OWN_CAR"])
        owns_realty = st.selectbox("Owns property?", category_map["FLAG_OWN_REALTY"])
        income_type = st.selectbox("Income type", category_map["NAME_INCOME_TYPE"])

    st.subheader("Additional information")

    col3, col4 = st.columns(2)
    with col3:
        education = st.selectbox(
            "Education", category_map["NAME_EDUCATION_TYPE"]
        )
        family_status = st.selectbox(
            "Family status", category_map["NAME_FAMILY_STATUS"]
        )
        housing_type = st.selectbox(
            "Housing type", category_map["NAME_HOUSING_TYPE"]
        )

    with col4:
        occupation = st.selectbox(
            "Occupation", category_map["OCCUPATION_TYPE"]
        )
        age_group = st.selectbox("Age group", category_map["AGE_GROUP"])
        income_group = st.selectbox(
            "Income group", category_map["INCOME_GROUP"]
        )

    st.subheader("Contact information")
    col5, col6, col7 = st.columns(3)
    with col5:
        work_phone = st.selectbox("Work phone", [0, 1])
    with col6:
        phone = st.selectbox("Personal phone", [0, 1])
    with col7:
        email = st.selectbox("Email", [0, 1])

    submitted = st.form_submit_button("Predict credit risk", type="primary")

if submitted:
    is_employed = 1 if employed == "Yes" else 0
    employment_years_value = float(employment_years) if is_employed else float("nan")

    applicant = pd.DataFrame([{
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": owns_car,
        "FLAG_OWN_REALTY": owns_realty,
        "CNT_CHILDREN": int(children),
        "AMT_INCOME_TOTAL": float(annual_income),
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing_type,
        "FLAG_WORK_PHONE": int(work_phone),
        "FLAG_PHONE": int(phone),
        "FLAG_EMAIL": int(email),
        "OCCUPATION_TYPE": occupation,
        "CNT_FAM_MEMBERS": int(family_members),
        "AGE": float(age),
        "IS_EMPLOYED": is_employed,
        "EMPLOYMENT_YEARS": employment_years_value,
        "AGE_GROUP": age_group,
        "INCOME_GROUP": income_group,
    }])

    # Preserve exactly the training column order.
    expected_columns = numerical_cols + categorical_cols
    applicant = applicant[expected_columns]

    applicant[numerical_cols] = num_imputer.transform(applicant[numerical_cols])
    applicant[categorical_cols] = cat_imputer.transform(applicant[categorical_cols])

    applicant_processed = encoder.transform(applicant)
    risk_probability = float(model.predict_proba(applicant_processed)[0, 1])
    prediction = int(risk_probability >= 0.5)

    st.subheader("Prediction result")
    st.metric("Estimated probability of higher-risk class", f"{risk_probability:.1%}")

    if prediction == 1:
        st.error("Model prediction: Higher-risk class (1)")
    else:
        st.success("Model prediction: Lower-risk class (0)")

    st.caption(
        "The 0.5 threshold matches the threshold used for the final test evaluation. "
        "This probability is a model score, not a verified probability of loan default."
    )
