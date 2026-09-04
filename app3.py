import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import os

# Configuration
st.set_page_config(
    page_title="Stunting Risk Prediction System",
    # page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


#  Setup Cascading Style (CSS)

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

# Load Model & Define features model
@st.cache_resource
def load_model():
    artifact = joblib.load("stunting_xgb.pkl")
    return artifact

artifact = load_model()
xgb_model = artifact["model"]
le_ts = artifact["label_encoder"]
feature_columns = artifact["feature_columns"]

features_ts = [
    "measurement_age_months","sex","history_ari","wealth_index",
    "history_diarrhea","complete_immunization","maternal_education",
    "weight_lag_1","height_lag_1","bmi_lag_1",
    "stunting_lag_1"
]

# 5. SHAP EXPLAINER
@st.cache_resource
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_shap_explainer(xgb_model)

# 6. FUNCTION PREPROCESS DATA BARU
def preprocess_new_data(input_data,feature_columns):
    # One-hot encoding
    X_new = pd.get_dummies(input_data,drop_first=True)
    # Samakan feature dengan training
    X_new = X_new.reindex(columns=feature_columns,fill_value=0)
    return X_new

# 7. FUNCTION SHAP INDIVIDUAL
def get_shap_values(model,explainer,X):
    shap_result = explainer(X)
    return shap_result

# 8. HEADER
st.markdown("""
<div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    padding: 28px;
    border-radius: 16px;
    color: white;
    margin-bottom: 24px;
">
<h1 style="margin:0;font-size:2.1rem;font-weight:700;">Stunting Risk Prediction Engine</h1>
<p style="margin-top:6px;opacity:0.85;font-size:0.95rem;">
Sistem Pendukung Keputusan Berbasis Machine Learning
untuk Pemetaan dan Intervensi Dini Risiko Stunting.
</p>
</div>
""", unsafe_allow_html=True)
# 9. TABS
tab1, tab2, tab3 = st.tabs(["📊 Overview Dataset","📋 Asesmen & Prediksi Individu","📊 Overview Model"])

# TAB 1
with tab1 :
    st.subheader("Ringkasan Dataset")
    # st.caption("Dataset yang digunakan ialah dataset sintetis.")
    st.text("Dataset ini merupakan dataset dummy/sintetis yang dibuat untuk keperluan pembelajaran dan pengembangan"
    "model Machine Learning dalam analisis risiko stunting pada anak. Dataset merepresentasikan data karakteristik anak,"
    "pengukuran antropometri, riwayat kesehatan, status imunisasi serta informasi kesehatan maternal yang berkaitan dengan"
    "kondisi pertumbuhan anak\n\n"
    
    "Setiap baris pada dataset merepresentasikan satu anak, sedangkan setiap kolom merepresentasikan karakteristik atau indikator"
    "kesehatan yang diamati. Variabel antropometri meliputi usia anak dalam bulan, berat badan, tinggi badan, dan BMI." 
    "Dataset juga mencakup beberapa variabel riwayat kesehatan seperti Acute Respiratory Infection (ARI),"
    "kelengkapan imunisasi, serta jumlah kunjungan Antenatal Care (ANC) selama masa kehamilan."

    "Variabel status_stunting digunakan sebagai variabel target yang menggambarkan kategori status pertumbuhan anak, misalnya Normal,"
    "Stunted, Severely Stunted, dan Tall. Karena dataset ini bersifat dummy/sintetis, data tidak merepresentasikan individu atau "
    "kondisi kesehatan nyata dan tidak dapat digunakan untuk diagnosis klinis maupun pengambilan keputusan medis.\n\n"

    "Dataset ditujukan untuk simulasi proses Exploratory Data Analysis (EDA), statistical analysis, feature selection, classification" 
    "modelling, model evaluation dan interpretasi risiko stunting",text_alignment="justify")
    
    st.subheader("Visualisasi")
    pic_1, caption_1 = st.columns(2)
    with pic_1 : 
        tile1 = st.container()
        tile1.image("grafik_kombinasi.png")
    with caption_1 :
        tile1 = st.container()
        tile1.text(
        "Wrap Grafik\n\n"
        "Sebagai studi awal, distribusi status pertumbuhan pada 121.902 pengukuran anak menunjukkan bahwa sebagian besar anak berada "
        "pada\nKategori Normal : 54.789 anak; Severely stunted : 28.868 anak; Tall 28.192 anak; Stunted 10.053 anak."
        "\nHal ini menunjukkan bahwa karakteristik "
        "status pertumbuhan cukup beragam, dengan kategori Stunted memiliki jumlah observasi paling rendah dibandingkan "
        "kategori lainnya\n"
        
        "Proses eksplorasi juga menunjukkan bahwa : \nStatus pertumbuhan anak memiliki hubungan yang kuat dengan karakteristik; "
        "antropometri dan umur.\nPola distribusi TB/U memperlihatkan pemisahan yang cukup jelas antar status; "
        "pertumbuhan TB/U dan HAZ menunjukkan adanya keterkaitan struktural antar indikator pertumbuhan."
        "\nTemuan ini mendukung penggunaan variabel antropometri dan riwayat pertumbuhan sebagai prediktor, tetapi juga "
        "menunjukkan perlunya perhatian terhadap redundansi informasi dan potensi leakage, khususnya apabila tujuan model adalah "
        "memprediksi insidensi stunting pada pengukuran berikutnya", text_alignment="justify")

    pic_2,caption_2 = st.columns(2)
    with pic_2 :
        tile2 = st.container()
        tile2.image("matrik_korelasi.png", caption="")
    with caption_2 :
        tile2 = st.container()
        tile2.text("Matrik Korelasi\n\nHasil analisis korelasi menunjukkan adanya hubungan yang cukup kuat antara\nBerat badan dengan BMI (r=0,73), "
        "\nTinggi badan dengan umur (r=0,69), \nTinggi badan dengan HAZ (r=0,69).\n\nKorelasi antara berat badan dan BMI merupakan "
        "konsekuensi dari hubungan matematis dalam perhitungan BMI, sedangkan korelasi tinggi badan dan umur mencerminkan pertumbuhan "
        "anak seiring bertambahnya usia. Korelasi tinggi badan dengan HAZ juga menunjukkan keterkaitan erat karena HAZ merupakan "
        "indikator tinggi badan menurut umur. Sebaliknya, sebagian besar variabel kelahiran dan tinggi badan ibu menunjukkan korelasi "
        "yang sangat rendah dengan antropometri anak pada saat pengukuran. Temuan ini menunjukkan bahwa variabel antropometri saat ini "
        "mengandung informasi yang kuat mengenai status pertumbuhan, namun juga memiliki potensi redundansi informasi sehingga perlu "
        "diperhatikan dalam pemilihan fitur dan interpretasi model",text_alignment="justify")
    
    pic_3,caption_3 = st.columns([2.5,0.5])
    with pic_3 :
        tile3 = st.container()
        tile3.image("alasan_time_series.png", caption="")
    with caption_3 :
        tile3 = st.container()
        tile3.text("Total anak: 24399\nAnak dengan status BERUBAH selama 5 bulan: 7389 (30.3%)\n"
        "Anak dengan status TETAP selama 5 bulan: 17010 (69.7%)",text_alignment="justify")

# TAB 2
with tab2:
    st.subheader("1. Masukkan Parameter Anak")
    st.caption(
        "Masukkan karakteristik anak, riwayat kesehatan, "
        "pertumbuhan, dan faktor sosial ekonomi."
    )

    # FORM
    with st.form("stunting_prediction_form"):        
        # PILAR 1        
        st.markdown("##### 👶 Demografi & Kondisi Kesehatan")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
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
        with col4:
            history_diarrhea = st.selectbox("Riwayat Diare",["No","Yes"])
        with col5:
            complete_immunization = st.selectbox("Imunisasi Lengkap",["Yes","No"])
        with col6:
            stunting_lag_1 = st.selectbox("Status Stunting Sebelumnya",["Normal","Severely stunted","Tall","Stunted"])

        # col4, col5, col6 = st.columns(3)
        # with col4:
        #     history_diarrhea = st.selectbox("Riwayat Diare",["No","Yes"])
        # with col5:
        #     complete_immunization = st.selectbox("Imunisasi Lengkap",["Yes","No"])
        # with col6:
        #     stunting_lag_1 = st.selectbox("Status Stunting Sebelumnya",["Normal","Severely stunted","Tall","Stunted"])
        st.markdown("---")

        
        # PILAR 2     
        st.markdown("##### 📏 Riwayat Pertumbuhan & Antropometri")
        col7, col8, col9 = st.columns(3)
        weight_lag_1 = col7.number_input(
            "Berat Badan Sebelumnya (kg)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1
        )
        height_lag_1 = col8.number_input(
            "Tinggi Badan Sebelumnya (cm)",
            min_value=1.0,
            max_value=200.0,
            value=80.0,
            step=0.1
        )
        # Hitung BMI otomatis
        if height_lag_1 > 0:
            bmi_lag_1 = weight_lag_1 / ((height_lag_1 / 100) ** 2)
        else:
            bmi_lag_1 = 0.0
        # Tampilkan BMI otomatis
        # col9.metric("BMI Sebelumnya",f"{bmi_lag_1:.2f}")
        col9.number_input(
            "BMI Sebelumnya",
            min_value=0.0,
            max_value=1000.0,
            value=float(bmi_lag_1),
            step=0.01,
            disabled=True
        )
        st.markdown("---")

        # PILAR 3
        st.markdown("##### 🔮 Growth Simulator")
        st.info(
            "Simulasikan perubahan status gizi berdasarkan perkiraan "
            "kenaikan berat dan tinggi badan dalam beberapa bulan ke depan."
        )

        col_sim1, col_sim2, col_sim3 = st.columns(3)

        with col_sim1:
            projection_months = st.number_input(
                "Periode Proyeksi (bulan)",
                min_value=1,
                max_value=24,
                value=2,
                step=1
            )

        with col_sim2:
            delta_weight = st.number_input(
                "Perkiraan Kenaikan BB (kg)",
                min_value=0.0,
                max_value=20.0,
                value=0.5,
                step=0.1
            )

        with col_sim3:
            delta_height = st.number_input(
                "Perkiraan Kenaikan TB (cm)",
                min_value=0.0,
                max_value=30.0,
                value=1.5,
                step=0.1
            )
        st.markdown("---")

        
        # PILAR 4
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

    # PREDICTION
    if btn_submit:
        # 0. GROWTH SIMULATOR
        ref_who = pd.read_csv("who_length_height_for_age_reference_0_60_months.xlsx - WHO_reference.csv")
        projected_age = measurement_age_months + projection_months
        projected_weight = weight_lag_1 + delta_weight
        projected_height = height_lag_1 + delta_height

        # haz future
        z = projected_height
        projected_age = int(round(projected_age))
        match = ref_who.loc[(ref_who["age_months"] == projected_age) & (ref_who["sex"] == sex)]
        if match.empty:
            np.nan
        ref = match.iloc[0]
        l_val, m_val, s_val = ref["L"], ref["M"], ref["S"]
        haz_future = (((z / m_val) ** l_val) - 1) / (l_val * s_val)
        haz_future = round(haz_future, 2)

        if pd.isna(haz_future):
            status_future = "Unknown"
        if haz_future < -3:
            status_future = "Severely stunted"
        elif haz_future < -2:
            status_future = "Stunted"
        elif haz_future <= 3:
            status_future = "Normal"
        else:
            status_future = "Tall"   

        # reverse haz
        if status_future in ["Severely stunted", "Stunted"]:
            haz_based = -1.99
            z_based = (m_val * ((haz_based * l_val * s_val) + 1) ** (1 / l_val))
        else:
            z_based = None

        # 1. CREATE DATAFRAME
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

        # 2. PREPROCESSING
        X_new = preprocess_new_data(input_data,feature_columns)
        # 3. PREDICTION
        prediction = xgb_model.predict(X_new)
        # 4. PREDICT PROBA
        probability = (xgb_model.predict_proba(X_new)[0])
        # 5. DECODE CLASS
        predicted_status = (le_ts.inverse_transform(prediction)[0])
        # 6. PROBABILITY DATAFRAME
        probability_df = pd.DataFrame({
            "Status":le_ts.classes_,
            "Probability":probability
        })
        probability_df["Probability (%)"] = (probability_df["Probability"]* 100)

        # 7. STUNTING RISK
        risk_classes = ["Stunted","Severely stunted"]
        risk_probability = (
            probability_df.loc[
                probability_df["Status"]
                .isin(risk_classes),
                "Probability"
            ].sum()
        )
        risk_percentage = (risk_probability * 100)

        # 8. RISK CATEGORY
        if risk_percentage >= 70:
            risk_category = ("RISIKO TINGGI")
        elif risk_percentage >= 30:
            risk_category = ("RISIKO SEDANG")
        else:
            risk_category = ("RISIKO RENDAH")
        
        st.markdown("---")

        st.subheader("2. Hasil Analisis Risiko")
        res_col1, res_col2, res_col3 = st.columns([0.8, 1.8, 1])

        # LEFT
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

        # RIGHT
        with res_col3:
            # Haz Now
            match = ref_who.loc[(ref_who["age_months"] == round(measurement_age_months)) & (ref_who["sex"] == sex)]
            if match.empty:
                np.nan
            ref = match.iloc[0]
            l_val, m_val, s_val = ref["L"], ref["M"], ref["S"]
            haz_now = (((height_lag_1 / m_val) ** l_val) - 1) / (l_val * s_val)
            haz_now = round(haz_now, 2)
            # Haz Now
            # st.markdown("##### 👶 Saat Ini")
            # st.write(f"**Umur**  {measurement_age_months:.0f} bulan")
            # st.write(f"**TB**  {height_lag_1:.1f} cm")
            # st.write(f"**HAZ**  {haz_now:.2f} SD")
            # st.write(f"**Status**  {predicted_status}")

            st.markdown(f"##### 📈 Proyeksi +{projection_months} Bulan")
            st.write(f"**Umur**  {projected_age:.0f} bulan")
            st.write(f"**TB**  {projected_height:.1f} cm  "
                f"(↑ {delta_height:.1f} cm)")
            st.write(f"**HAZ**  {haz_future:.2f} SD  "
                f"(↑ {haz_future - haz_now:.2f} SD)"
            )
        
            if status_future in ["Severely stunted", "Stunted"]:
                st.error(f"🚨 **{status_future}**")
            else:
                st.success(f"✅ **{status_future}**")

            if z_based is not None:
                st.markdown(f"##### 🎯 Target Naik TB {projection_months} Bulan (min)")
                st.write(f"**Umur**  {projected_age:.0f} bulan")
                st.write(f"**TB**  {z_based:.1f} cm  "
                    f"(↑ {z_based-projected_height:.1f} cm)")
                # st.write(f"**HAZ**  {haz_based:.2f} SD  ")


        # SHAP
        st.markdown("---")
        st.subheader("3. Analisis Faktor Risiko (SHAP)")
        st.caption(
            "SHAP menunjukkan kontribusi masing-masing "
            "fitur terhadap prediksi model."
        )
        
        # SHAP CALCULATION
        shap_result = get_shap_values(xgb_model,explainer,X_new)
        # IDENTIFY STUNTED CLASS
        classes = list(le_ts.classes_)
        stunted_class = classes.index("Stunted")
        severe_class = classes.index("Severely stunted")
        # SHAP VALUES
        shap_stunted = (shap_result.values[0,:,stunted_class])
        shap_severe = (shap_result.values[0,:,severe_class])
        # COMBINED SHAP
        shap_risk = (shap_stunted+shap_severe)
        # SHAP DATAFRAME
        shap_df = pd.DataFrame({"Feature":X_new.columns,"SHAP":shap_risk})
        shap_df["Absolute SHAP"] = (shap_df["SHAP"].abs())
        shap_df = shap_df.sort_values("Absolute SHAP",ascending=False)

        # TOP SHAP
        top_shap = (shap_df.head(10).sort_values("SHAP"))
        fig_shap = px.bar(top_shap,x="SHAP",y="Feature",orientation="h",text="SHAP")
        fig_shap.update_traces(texttemplate="%{text:.3f}",textposition="outside")
        fig_shap.update_layout(height=400,
            margin=dict(l=10,r=40,t=10,b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_shap,use_container_width=True)

        # TOP FACTORS
        factor_col1, factor_col2 = st.columns(2)
        # INCREASE RISK
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
        
        # DECREASE RISK
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
        # INPUT DATA
        st.markdown("---")
        st.markdown("##### 📋 Data Input")
        st.dataframe(
            input_data.T.rename(columns={0: "Nilai"}),
            use_container_width=True
        )

        # PROBABILITY TABLE
        st.markdown("##### 📊 Detail Probabilitas")
        st.dataframe(
            probability_df[["Status","Probability (%)"]]
            .style.format({"Probability (%)": "{:.2f}%"}),
            hide_index=True,
            use_container_width=True
        )

# TAB 3
with tab3:
    st.subheader("Evaluasi Performa Model Machine Learning : Model XGBoost")
    st.caption("Ringkasan validasi performa model Machine Learning yang digunakan oleh sistem.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Akurasi (Accuracy)", "81.21%", "+1.5%")
    m2.metric("ROC-AUC Score", "0.9263", "High Discrimination")
    m3.metric("F1-Score", "0.7343", "Balanced")

    # MODEL INFORMATION
    # col1, col2, col3, col4 = st.columns(4)
    # col1.metric("Model","XGBoost")
    # col2.metric("Features",len(features_ts))
    # col3.metric("Target","Multiclass")
    # col4.metric("Classes",len(le_ts.classes_))

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("##### Global Feature Importance")
        # Grafik Importance dengan Plotly Express
        global_feat = pd.DataFrame({
            'Indikator': [
                'stunting_lag_1_Tall', 'stunting_lag_1_Severely stunted', 'stunting_lag_1_Stunted', 
                'height_lag_1', 'measurement_age_months', 'sex', 'wealth_index',
                'history_diarrhea', 'history_ari', 'wealth_index'
            ],
            'Importance': [0.42,0.42,0.13,0.006,0.003,0.001,0.0005,0.0004,0.0004,0.0004]
        }).sort_values(by='Importance', ascending=True)
        
        fig_imp = px.bar(
            global_feat, 
            x='Importance', 
            y='Indikator', 
            orientation='h', 
            color_discrete_sequence=['#2563EB']
        )
        fig_imp.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col_chart2:
        st.markdown("##### Confusion Matrix")
        # Heatmap Confusion Matrix dengan Plotly Express
        conf_matrix = pd.DataFrame(
            [[9466,  249,  689,  728],
            [ 182, 4945,  577,    0],
            [ 642,  709,  789,    0],
            [ 800,    0,    0, 4573]], 
            columns=['Prediksi Normal', 'Prediksi Severely stunted', 'Prediksi Stunted', 'Prediksi Tall'],
            index=['Aktual Normal', 'Aktual Severely stunted', 'Aktual Stunted', 'Aktual Tall']
        )
        fig_cm = px.imshow(
            conf_matrix, 
            text_auto=True, 
            color_continuous_scale='Blues',
            labels=dict(x="Hasil Prediksi Model", y="Kondisi Asli (Aktual)", color="Jumlah Pasien")
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    
    desc1,desc2 = st.columns(2)
    with desc1 :
        # Class mapping
        st.markdown("##### Target Classes")
        class_df = pd.DataFrame({
            "Encoded":range(len(le_ts.classes_)),
            "Status":le_ts.classes_
        })
        st.dataframe(class_df,hide_index=True,use_container_width=True)
    with desc2 :
        # Features
        st.markdown("##### Model Features")
        feature_df = pd.DataFrame({
            "No":range(1,len(features_ts) + 1),
            "Feature":features_ts
        })
        st.dataframe(feature_df,hide_index=True,use_container_width=True)