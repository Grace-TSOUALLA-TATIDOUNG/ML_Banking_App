import time
import streamlit as st
import requests


BASE_API_URL = "https://bank-marketing-prediction-api.onrender.com"
HEALTH_URL = f"{BASE_API_URL}/health"
PREDICT_URL = f"{BASE_API_URL}/predict"


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


def wake_up_api(max_retries=3, wait_time=5):
    for _ in range(max_retries):
        try:
            response = requests.get(HEALTH_URL, timeout=30)

            if response.status_code == 200:
                health_data = response.json()

                if health_data.get("model_loaded") is True:
                    return True

        except requests.exceptions.RequestException:
            pass

        time.sleep(wait_time)

    return False


def call_api_with_retry(payload, max_retries=3, wait_time=5):
    response = None

    for _ in range(max_retries):
        response = requests.post(
            PREDICT_URL,
            json=payload,
            timeout=60
        )

        if response.status_code != 429:
            return response

        time.sleep(wait_time)

    return response


if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0


with st.form("prediction_form"):

    st.divider()
    st.subheader("Client information")

    age = st.number_input("Age", value=35)

    job = st.selectbox(
        "Job",
        [
            "admin.", "blue-collar", "entrepreneur", "housemaid",
            "management", "retired", "self-employed", "services",
            "student", "technician", "unemployed", "unknown"
        ]
    )

    marital = st.selectbox(
        "Marital status",
        ["divorced", "married", "single", "unknown"]
    )

    education = st.selectbox(
        "Education",
        [
            "basic.4y", "basic.6y", "basic.9y", "high.school",
            "illiterate", "professional.course", "university.degree", "unknown"
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

    duration = st.number_input("Last contact duration in seconds", value=180)

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

    emp_var_rate = st.number_input("Employment variation rate", value=1.1)
    cons_price_idx = st.number_input("Consumer price index", value=93.994)
    cons_conf_idx = st.number_input("Consumer confidence index", value=-36.4)
    euribor3m = st.number_input("Euribor 3 month rate", value=4.857)
    nr_employed = st.number_input("Number of employees", value=5191.0)

    submitted = st.form_submit_button("Predict subscription")


if submitted:
    cooldown = 10
    now = time.time()

    if now - st.session_state.last_request_time < cooldown:
        st.warning("Please wait a few seconds before trying again.")
        st.stop()

    st.session_state.last_request_time = now

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

    try:
        with st.spinner("Starting prediction service..."):
            api_ready = wake_up_api()

        if not api_ready:
            st.warning(
                "The prediction service is starting up. "
                "Please wait a few seconds and try again."
            )
            st.stop()

        with st.spinner("Prediction in progress..."):
            response = call_api_with_retry(payload)

        if response is None:
            st.error("The prediction service did not return a response.")
            st.stop()

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
            st.warning(
                "The prediction service is temporarily busy. "
                "Please wait a few seconds and try again."
            )

        elif response.status_code == 503:
            st.warning(
                "The prediction model is still loading. "
                "Please wait a few seconds and try again."
            )

        else:
            st.error("The prediction service returned an error.")
            st.write(f"Status code: {response.status_code}")

    except requests.exceptions.Timeout:
        st.error("The API took too long to respond. Please try again.")

    except requests.exceptions.RequestException:
        st.error(
            "Unable to connect to the prediction API. "
            "The service may be starting up."
        )