import streamlit as st
import pandas as pd
import numpy as np
import joblib


clf = joblib.load("classifier.pkl")
reg = joblib.load("regressor.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")


st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🚗 EV Thermal Management Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")


df = pd.read_csv("ev_thermal_mixed_dataset.csv")


st.markdown("### 📊 Dataset Preview")
st.dataframe(df.head())


numeric_cols = ['time_s','ambient_temp','current_A','SOC_percent','system_temp']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].median(), inplace=True)

categorical_cols = ['voltage_V','coolant_flow','fan_speed']
for col in categorical_cols:
    df[col] = df[col].astype(str)
    df[col].replace("nan", np.nan, inplace=True)
    df[col].fillna(df[col].mode()[0], inplace=True)

encoded = encoder.transform(df[categorical_cols]).toarray()
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

X = pd.concat([df[numeric_cols], encoded_df], axis=1)
X_scaled = scaler.transform(X)


def control_strategy(pred_temp, pred_risk):
    safe_min, safe_max = 25, 45
    if pred_temp > safe_max or pred_risk == 1:
        return {"coolant_flow":"strong", "fan_speed":"fast", "heating":"off"}
    elif pred_temp < safe_min:
        return {"coolant_flow":"weak", "fan_speed":"slow", "heating":"on"}
    else:
        return {"coolant_flow":"normal", "fan_speed":"medium", "heating":"off"}


st.markdown("### ⚙️ Control Strategy Tester")
sample_idx = st.slider("Pick a sample index", 0, len(X_scaled)-1, 5)

sample_features = X_scaled[sample_idx].reshape(1,-1)
pred_risk = clf.predict(sample_features)[0]
pred_temp = reg.predict(sample_features)[0]
decision = control_strategy(pred_temp, pred_risk)


st.markdown(f"<h3 style='color: #27AE60;'>Predicted Temp: {pred_temp:.2f} °C</h3>", unsafe_allow_html=True)
st.markdown(f"<h3 style='color: #C0392B;'>Predicted Risk: {pred_risk}</h3>", unsafe_allow_html=True)

st.markdown("### 🛠️ Suggested Control Decision")
st.json(decision)
