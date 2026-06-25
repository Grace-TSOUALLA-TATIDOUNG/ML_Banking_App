import streamlit as st
import requests


API_URL = "https://bank-marketing-prediction-api.onrender.com/predict"

st.set_page_config(
    page_title="Bank Marketing Prediction App",
    page_icon="🏦",
    layout="centered"
)

st.title("Bank Marketing Prediction App")

st.write(
    "This application predicts whether a bank client is likely to subscribe "
    "to a term deposit based on client profile, campaign and economic data."
)

st.divider()

st.subheader("Client information")

age = st.number_input("Age", value=35)

job = st.selectbox(
    "Job",
    [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown"
    ]
)

marital = st.selectbox(
    "Marital status",
    ["divorced", "married", "single", "unknown"]
)

education = st.selectbox(
    "Education",
    [
        "basic.4y",
        "basic.6y",
        "basic.9y",
        "high.school",
        "illiterate",
        "professional.course",
        "university.degree",
        "unknown"
    ]
)

default = st.selectbox("Has credit in default?", ["no", "yes", "unknown"])
housing = st.selectbox("Has housing loan?", ["no", "yes", "unknown"])
loan = st.selectbox("Has personal loan?", ["no", "yes", "unknown"])

st.divider()

st.subheader("Campaign information")

contact = st.selectbox("Contact communication type", ["cellular", "telephone"])

month = st.selectbox(
    "Last contact month",
    ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
)

day_of_week = st.selectbox(
    "Last contact day of the week",
    ["mon", "tue", "wed", "thu", "fri"]
)

duration = st.number_input(
    "Last contact duration in seconds",
    value=180
)

campaign = st.number_input(
    "Number of contacts during this campaign",
    value=2
)

pdays = st.number_input(
    "Days since the client was last contacted (999 if not)",
    value=999
)

previous = st.number_input(
    "Number of contacts before this campaign",
    value=0
)

poutcome = st.selectbox(
    "Outcome of the previous campaign",
    ["failure", "nonexistent", "success"]
)

st.divider()

st.subheader("Economic context")

emp_var_rate = st.number_input(
    "Employment variation rate",
    value=1.1
)

cons_price_idx = st.number_input(
    "Consumer price index",
    value=93.994
)

cons_conf_idx = st.number_input(
    "Consumer confidence index",
    value=-36.4
)

euribor3m = st.number_input(
    "Euribor 3 month rate",
    value=4.857
)

nr_employed = st.number_input(
    "Number of employees",
    value=5191.0
)

st.divider()

payload = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "month": month,
    "day_of_week": day_of_week,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome,
    "emp.var.rate": emp_var_rate,
    "cons.price.idx": cons_price_idx,
    "cons.conf.idx": cons_conf_idx,
    "euribor3m": euribor3m,
    "nr.employed": nr_employed
}

if "is_loading" not in st.session_state:
    st.session_state.is_loading = False

if st.button("Predict subscription", disabled=st.session_state.is_loading):
    st.session_state.is_loading = True

    try:
        with st.spinner("Prediction in progress..."):
            response = requests.post(
                API_URL,
                json=payload,
                timeout=60
            )

        st.write("Status code:", response.status_code)
        st.write("Response text:", response.text)
        
        if response.status_code == 200:
            result = response.json()

            prediction = result["prediction"]
            probability = result["probability"]

            st.subheader("Prediction result")

            if prediction == "yes":
                st.success("The client is likely to subscribe to a term deposit.")
            else:
                st.warning("The client is unlikely to subscribe to a term deposit.")

            st.write(f"**Prediction:** {prediction}")
            st.write(f"**Probability:** {probability:.2%}")

        elif response.status_code == 429:
            st.error("Too many requests. Please wait a few seconds and try again.")
            st.write(response.text)

        else:
            st.error(f"The API returned an error: {response.status_code}")
            st.write(response.text)

    except requests.exceptions.Timeout:
        st.error("The API took too long to respond. Please try again.")

    except requests.exceptions.RequestException as e:
        st.error("Unable to connect to the API.")
        st.write(str(e))

    finally:
        st.session_state.is_loading = False