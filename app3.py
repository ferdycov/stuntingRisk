import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import os


# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Stunting Risk Prediction System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #F8FAFC;
    font-family: 'Inter', sans-serif;
}

div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

div[data-testid="stForm"] {
    background-color: #FFFFFF;
    border-radius: 16px;
    padding: 24px;
    border: 1px solid #E2E8F0;
}

.stButton>button {
    width: 100%;
    background-color: #2563EB;
    color: white;
    font-weight: 600;
    padding: 12px 24px;
    border-radius: 10px;
    border: none;
}

.stButton>button:hover {
    background-color: #1D4ED8;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    artifact = joblib.load("stunting_xgb.pkl")
    return artifact

artifact = load_model()
xgb_model = artifact["model"]
le_ts = artifact["label_encoder"]
feature_columns = artifact["feature_columns"]

# =========================================================
# 4. MODEL FEATURES
# =========================================================

features_ts = [
    "measurement_age_months",
    "sex",
    "history_ari",
    "history_diarrhea",
    "complete_immunization",
    "weight_lag_1",
    "height_lag_1",
    "bmi_lag_1",
    "stunting_lag_1",
    "wealth_index",
    "maternal_education"
]


# =========================================================
# 5. SHAP EXPLAINER
# =========================================================
@st.cache_resource
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_shap_explainer(xgb_model)
# =========================================================
# 6. FUNCTION PREPROCESS DATA BARU
# =========================================================
def preprocess_new_data(input_data,feature_columns):
    # One-hot encoding
    X_new = pd.get_dummies(input_data,drop_first=True)
    # Samakan feature dengan training
    X_new = X_new.reindex(columns=feature_columns,fill_value=0)
    return X_new

# =========================================================
# 7. FUNCTION SHAP INDIVIDUAL
# =========================================================
def get_shap_values(model,explainer,X):
    shap_result = explainer(X)
    return shap_result
# =========================================================
# 8. HEADER
# =========================================================
st.markdown("""
<div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    padding: 28px;
    border-radius: 16px;
    color: white;
    margin-bottom: 24px;
">
<h1 style="margin:0;font-size:2.1rem;font-weight:700;">🛡️ Stunting Risk Prediction Engine</h1>
<p style="margin-top:6px;opacity:0.85;font-size:0.95rem;">
Sistem Pendukung Keputusan Berbasis Machine Learning
untuk Pemetaan dan Intervensi Dini Risiko Stunting.
</p>
</div>
""", unsafe_allow_html=True)
# =========================================================
# 9. TABS
# =========================================================
tab1, tab2 = st.tabs(["📋 Asesmen & Prediksi Individu","📊 Overview Model"])

# =========================================================
# TAB 1
# =========================================================

with tab1:
    st.subheader("1. Masukkan Parameter Anak")
    st.caption(
        "Masukkan karakteristik anak, riwayat kesehatan, "
        "pertumbuhan, dan faktor sosial ekonomi."
    )

    # =====================================================
    # FORM
    # =====================================================

    with st.form("stunting_prediction_form"):
        # -------------------------------------------------
        # PILAR 1
        # -------------------------------------------------
        st.markdown("##### 👶 Demografi & Kondisi Kesehatan")

        col1, col2, col3 = st.columns(3)
        with col1:
            measurement_age_months = st.number_input(
                "Usia Pengukuran (bulan)",
                min_value=0.0,
                max_value=120.0,
                value=24.0,
                step=1.0
            )

        with col2:
            sex = st.selectbox("Jenis Kelamin",["Female","Male"])

        with col3:
            history_ari = st.selectbox("Riwayat ISPA",["No","Yes"])

        col4, col5, col6 = st.columns(3)
        with col4:
            history_diarrhea = st.selectbox("Riwayat Diare",["No","Yes"])

        with col5:
            complete_immunization = st.selectbox("Imunisasi Lengkap",["Yes","No"])

        with col6:
            stunting_lag_1 = st.selectbox("Status Stunting Sebelumnya",["Normal","Severely stunted","Tall","Stunted"])

        st.markdown("---")

        # -------------------------------------------------
        # PILAR 2
        # -------------------------------------------------

        st.markdown("##### 📏 Riwayat Pertumbuhan & Antropometri")

        col7, col8, col9 = st.columns(3)
        with col7:
            weight_lag_1 = st.number_input(
                "Berat Badan Sebelumnya (kg)",
                min_value=0.0,
                max_value=100.0,
                value=8.5,
                step=0.1
            )

        with col8:
            height_lag_1 = st.number_input(
                "Tinggi Badan Sebelumnya (cm)",
                min_value=0.0,
                max_value=200.0,
                value=78.0,
                step=0.1
            )

        with col9:
            bmi_lag_1 = st.number_input(
                "BMI Sebelumnya",
                min_value=0.0,
                max_value=50.0,
                value=13.9,
                step=0.1
            )

        st.markdown("---")


        # -------------------------------------------------
        # PILAR 3
        # -------------------------------------------------

        st.markdown("##### 🏡 Sosio-Ekonomi & Pendidikan")

        col10, col11 = st.columns(2)

        with col10:
            wealth_index = st.selectbox("Indeks Kesejahteraan",
                ["Poorest","Poorer","Middle","Richer","Richest"])

        with col11:
            maternal_education = st.selectbox("Pendidikan Ibu",
                ["Primary school","Junior high school","Senior high school","Bachelors degree"])

        st.markdown( "<br>",unsafe_allow_html=True)

        btn_submit = st.form_submit_button("🚀 Jalankan Prediksi Risiko")

    # =====================================================
    # PREDICTION
    # =====================================================

    if btn_submit:
        # =================================================
        # 1. CREATE DATAFRAME
        # =================================================
        input_data = pd.DataFrame([{
            "measurement_age_months":measurement_age_months,
            "sex":sex,
            "history_ari":history_ari,
            "history_diarrhea":history_diarrhea,
            "complete_immunization":complete_immunization,
            "weight_lag_1":weight_lag_1,
            "height_lag_1":height_lag_1,
            "bmi_lag_1":bmi_lag_1,
            "stunting_lag_1":stunting_lag_1,
            "wealth_index":wealth_index,
            "maternal_education":maternal_education
        }])

        # =================================================
        # 2. PREPROCESSING
        # =================================================
        X_new = preprocess_new_data(input_data,feature_columns)

        # =================================================
        # 3. PREDICTION
        # =================================================
        prediction = xgb_model.predict(X_new)

        # =================================================
        # 4. PREDICT PROBA
        # =================================================
        probability = (xgb_model.predict_proba(X_new)[0])

        # =================================================
        # 5. DECODE CLASS
        # =================================================
        predicted_status = (le_ts.inverse_transform(prediction)[0])

        # =================================================
        # 6. PROBABILITY DATAFRAME
        # =================================================
        probability_df = pd.DataFrame({
            "Status":le_ts.classes_,
            "Probability":probability
        })

        probability_df["Probability (%)"] = (probability_df["Probability"]* 100)

        # =================================================
        # 7. STUNTING RISK
        # =================================================

        risk_classes = ["Stunted","Severely stunted"]
        risk_probability = (
            probability_df.loc[
                probability_df["Status"]
                .isin(risk_classes),
                "Probability"
            ].sum()
        )

        risk_percentage = (risk_probability * 100)

        # =================================================
        # 8. RISK CATEGORY
        # =================================================

        if risk_percentage >= 70:
            risk_category = ("RISIKO TINGGI")
        elif risk_percentage >= 30:
            risk_category = ("RISIKO SEDANG")
        else:
            risk_category = ("RISIKO RENDAH")

        # =================================================
        # RESULT
        # =================================================
        st.markdown("---")

        st.subheader("2. Hasil Analisis Risiko")

        res_col1, res_col2 = st.columns([1, 1.8])

        # =================================================
        # LEFT
        # =================================================

        with res_col1:

            st.markdown("##### Indikator Risiko")
            st.metric("Probabilitas Risiko Stunting",f"{risk_percentage:.1f}%")
            st.progress(float(min(risk_percentage / 100, 1.0)))

            if risk_percentage >= 70:
                st.error(f"🚨 **{risk_category}**")
            elif risk_percentage >= 30:
                st.warning(f"⚠️ **{risk_category}**")
            else:
                st.success(f"✅ **{risk_category}**")

            st.markdown("##### Prediksi Status")
            st.info(f"**{predicted_status}**")

        # =================================================
        # RIGHT
        # =================================================

        with res_col2:
            st.markdown("##### Probabilitas Setiap Status")

            fig_prob = px.bar(probability_df,
                x="Probability (%)",
                y="Status",
                orientation="h",
                text="Probability (%)"
            )

            fig_prob.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
            fig_prob.update_layout(
                height=300,
                margin=dict(l=10,r=20,t=10,b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig_prob,use_container_width=True)

        # =================================================
        # SHAP
        # =================================================

        st.markdown("---")

        st.subheader("3. Analisis Faktor Risiko (SHAP)")

        st.caption(
            "SHAP menunjukkan kontribusi masing-masing "
            "fitur terhadap prediksi model."
        )

        # -------------------------------------------------
        # SHAP CALCULATION
        # -------------------------------------------------
        shap_result = get_shap_values(xgb_model,explainer,X_new)

        # =================================================
        # IDENTIFY STUNTED CLASS
        # =================================================
        classes = list(le_ts.classes_)
        stunted_class = classes.index("Stunted")
        severe_class = classes.index("Severely stunted")
        # =================================================
        # SHAP VALUES
        # =================================================
        shap_stunted = (shap_result.values[0,:,stunted_class])
        shap_severe = (shap_result.values[0,:,severe_class])

        # =================================================
        # COMBINED SHAP
        # =================================================
        shap_risk = (shap_stunted+shap_severe)

        # =================================================
        # SHAP DATAFRAME
        # =================================================
        shap_df = pd.DataFrame({"Feature":X_new.columns,"SHAP":shap_risk})
        shap_df["Absolute SHAP"] = (shap_df["SHAP"].abs())
        shap_df = shap_df.sort_values("Absolute SHAP",ascending=False)

        # =================================================
        # TOP SHAP
        # =================================================
        top_shap = (shap_df.head(10).sort_values("SHAP"))
        fig_shap = px.bar(top_shap,x="SHAP",y="Feature",orientation="h",text="SHAP")
        fig_shap.update_traces(texttemplate="%{text:.3f}",textposition="outside")
        fig_shap.update_layout(height=400,
            margin=dict(l=10,r=40,t=10,b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig_shap,use_container_width=True)

        # =================================================
        # TOP FACTORS
        # =================================================
        factor_col1, factor_col2 = st.columns(2)
        # -------------------------------------------------
        # INCREASE RISK
        # -------------------------------------------------
        with factor_col1:
            st.markdown("##### 🔴 Faktor yang Meningkatkan Risiko")

            risk_factors = (
                shap_df[shap_df["SHAP"] > 0].sort_values("SHAP",ascending=False).head(5)
            )

            if len(risk_factors) > 0:
                for _, row in risk_factors.iterrows():
                    st.write(
                        f"**{row['Feature']}**  "
                        f"+{row['SHAP']:.3f}"
                    )
            else:
                st.success("Tidak terdapat faktor positif dominan.")
        # -------------------------------------------------
        # DECREASE RISK
        # -------------------------------------------------

        with factor_col2:

            st.markdown("##### 🟢 Faktor yang Menurunkan Risiko")
            protective_factors = (
                shap_df[shap_df["SHAP"] < 0].sort_values("SHAP").head(5)
            )

            if len(protective_factors) > 0:
                for _, row in protective_factors.iterrows():
                    st.write(
                        f"**{row['Feature']}**  "
                        f"{row['SHAP']:.3f}"
                    )
            else:
                st.info("Tidak terdapat faktor protektif dominan.")
        # =================================================
        # INPUT DATA
        # =================================================
        st.markdown("---")
        st.markdown("##### 📋 Data Input")
        st.dataframe(
            input_data.T.rename(columns={0: "Nilai"}),
            use_container_width=True
        )

        # =================================================
        # PROBABILITY TABLE
        # =================================================
        st.markdown("##### 📊 Detail Probabilitas")
        st.dataframe(
            probability_df[
                [
                    "Status",
                    "Probability (%)"
                ]
            ].style.format({
                "Probability (%)": "{:.2f}%"
            }),
            hide_index=True,
            use_container_width=True
        )


# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.subheader("Evaluasi Model XGBoost")
    st.caption("Informasi model yang digunakan oleh sistem.")

    # =====================================================
    # MODEL INFORMATION
    # =====================================================
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model","XGBoost")
    col2.metric("Features",len(features_ts))
    col3.metric("Target","Multiclass")
    col4.metric("Classes",len(le_ts.classes_))
    st.markdown("---")

    # =====================================================
    # CLASS MAPPING
    # =====================================================
    st.markdown("##### Target Classes")
    class_df = pd.DataFrame({
        "Encoded":range(len(le_ts.classes_)),
        "Status":le_ts.classes_
    })
    st.dataframe(class_df,hide_index=True,use_container_width=True)

    # =====================================================
    # FEATURES
    # =====================================================
    st.markdown("##### Model Features")
    feature_df = pd.DataFrame({
        "No":range(1,len(features_ts) + 1),
        "Feature":features_ts
    })
    st.dataframe(feature_df,hide_index=True,use_container_width=True)