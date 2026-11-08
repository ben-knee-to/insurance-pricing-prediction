"""Run: streamlit run app.py"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.express as px

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # required for loading the custom pipeline transformer
from prepare_features import FEATURES
from app_utils import validate_customer, get_concentration, predict_charge

st.set_page_config(page_title="Insurance Charge Analytics", page_icon="📊", layout="wide")
st.title("Insurance Charge Prediction & Analytics")
st.caption("Educational portfolio analysis · US medical charges in USD · Not an insurance quote or underwriting decision")

@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data/processed/insurance_clean.csv")

@st.cache_resource
def load_model():
    path = ROOT / "models/final_insurance_model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)  # only load locally generated, trusted artifacts

try:
    df = load_data()
except Exception as exc:
    st.error(f"Cleaned dataset missing or unreadable: {exc}. Run python src/clean_data.py")
    st.stop()

pages = st.tabs(["Overview", "Customer analytics", "Cost concentration", "Predict charges", "Model performance", "Explainability"])
with pages[0]:
    st.subheader("Dataset overview")
    a,b,c,d=st.columns(4)
    a.metric("Customers",f"{len(df):,}")
    b.metric("Mean charges",f"${df.charges.mean():,.0f}")
    c.metric("Median charges",f"${df.charges.median():,.0f}")
    d.metric("Smokers",f"{(df.smoker=='yes').mean():.1%}")
    st.plotly_chart(px.histogram(df,x="charges",nbins=45,title="Medical charge distribution",labels={"charges":"Charges (USD)"}),use_container_width=True)
    st.dataframe(df.head(12),hide_index=True)
with pages[1]:
    st.subheader("Customer segments")
    group=st.selectbox("Compare charges by",["smoker","region","sex","children"],key="group")
    summary=df.groupby(group,dropna=False).charges.agg(count="count",mean="mean",median="median").reset_index()
    st.dataframe(summary,hide_index=True)
    st.plotly_chart(px.bar(summary,x=group,y="mean",title="Average charges by group",labels={"mean":"Mean charges (USD)"}),use_container_width=True)
    st.plotly_chart(px.scatter(df,x="bmi",y="charges",color="smoker",hover_data=["age"],title="BMI, smoking and charges"),use_container_width=True)
    df2=df.copy(); df2["age_band"]=pd.cut(df2.age,[0,25,35,45,55,120],labels=["≤25","26–35","36–45","46–55","56+"])
    st.plotly_chart(px.box(df2,x="age_band",y="charges",color="smoker",title="Charges by age band and smoking"),use_container_width=True)
with pages[2]:
    st.subheader("Charge concentration")
    c=get_concentration(df.charges)
    a,b,c3=st.columns(3)
    a.metric("Top 10% share",f"{c['top_10_share']:.1%}")
    b.metric("Top 20% share",f"{c['top_20_share']:.1%}")
    c3.metric("Gini coefficient",f"{c['gini']:.3f}")
    st.plotly_chart(px.line(pd.DataFrame({"Population share":c["population"],"Charges share":c["cumulative"]}),x="Population share",y="Charges share",title="Lorenz curve"),use_container_width=True)
    st.caption("Charges are not claims, premiums or insurer losses; concentration describes this dataset only.")
with pages[3]:
    st.subheader("Estimate medical charges")
    with st.form("prediction"):
        a,b,c=st.columns(3)
        age=a.number_input("Age",18,64,39)
        bmi=b.number_input("BMI",15.0,55.0,30.0,step=0.1)
        children=c.number_input("Children",0,5,0)
        sex=a.selectbox("Sex",["female","male"])
        smoker=b.selectbox("Smoker",["no","yes"])
        region=c.selectbox("Region",["northeast","northwest","southeast","southwest"])
        submitted=st.form_submit_button("Predict medical charges")
    if submitted:
        model=load_model()
        if model is None:
            st.error("Model file not found. Run python src/tune_models.py to generate it locally.")
        else:
            customer={"age":age,"sex":sex,"bmi":bmi,"children":children,"smoker":smoker,"region":region}
            try:
                amount=predict_charge(model,customer)
                st.metric("Estimated charges (USD)",f"${amount:,.2f}")
                st.info("This is a statistical estimate for a synthetic dataset, not a medical assessment, premium quote, or guaranteed amount.")
            except (ValueError,TypeError) as exc:
                st.error(str(exc))
with pages[4]:
    st.subheader("Frozen model evaluation")
    path=ROOT/"reports/ml/step6/final_evaluation.json"
    if path.exists():
        info=__import__('json').loads(path.read_text())
        m=info['holdout_metrics']; a,b,c=st.columns(3)
        a.metric("Test MAE",f"${m['mae']:,.0f}")
        b.metric("Test RMSE",f"${m['rmse']:,.0f}")
        c.metric("Test R²",f"{m['r2']:.3f}")
    f=ROOT/"reports/ml/step6/final_holdout_predictions.csv"
    if f.exists():
        pred=pd.read_csv(f)
        st.dataframe(pred.head(20),hide_index=True)
    for file in ["holdout_actual_vs_predicted.png","holdout_residuals.png"]:
        fig=ROOT/"reports/ml/step6/figures"/file
        if fig.exists(): st.image(str(fig),use_container_width=True)
    st.warning("The test set was used once for final evaluation. Further model selection using it would invalidate its status as an untouched holdout.")
with pages[5]:
    st.subheader("Model explanations")
    st.write("These explanations describe statistical model behavior, not causal effects.")
    for file in ["permutation_importance.png","shap_global_importance.png","segment_errors.png"]:
        fig=ROOT/"reports/ml/step7/figures"/file
        if fig.exists(): st.image(str(fig),use_container_width=True)
    f=ROOT/"reports/ml/step7/shap_local_training_example.csv"
    if f.exists():
        st.markdown("#### SHAP contributions for one training example")
        st.dataframe(pd.read_csv(f),hide_index=True)
    st.caption("Local SHAP example is from a training record; it is not an explanation of the form prediction above.")
