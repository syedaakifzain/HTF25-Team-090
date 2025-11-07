import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import subprocess, sys, time, io
from pathlib import Path

# Page config
st.set_page_config(page_title="Asteroid Classifier", page_icon="🌌", layout="wide", initial_sidebar_state="expanded")

PRIMARY, SUCCESS, WARNING, DANGER = "#2ba8a8", "#28a745", "#ff9800", "#dc3545"
CSS = f"""
<style>
:root {{ --primary:{PRIMARY}; --success:{SUCCESS}; --warning:{WARNING}; --danger:{DANGER}; }}
.metric-card {{ background:#0e1117; padding:12px 16px; border-radius:12px; border:1px solid #2b2b2b; }}
.metric-title {{ font-size:12px; opacity:.8; }} .metric-value {{ font-size:22px; font-weight:700; }}
.red {{ color:{DANGER}; }} .green {{ color:{SUCCESS}; }} .orange {{ color:{WARNING}; }} .teal {{ color:{PRIMARY}; }}
.section {{ border-left:4px solid var(--primary); padding-left:12px; margin:12px 0; }}
.btn-primary {{ background: var(--primary); color:white; padding:8px 14px; border-radius:8px; }}
.dark-bg {{ background:#0e1117; color:#e1e1e6; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    logo_candidates = [Path("logo.png"), Path("assets/logo.png"), Path("presentation/logo.png")] 
    logo_path = next((p for p in logo_candidates if p.exists()), None)
    if logo_path:
        st.image(str(logo_path), width=200)
    else:
        st.markdown("### 🌌 Asteroid Classifier")
    st.markdown("### 🌌 Asteroid Hazard Classification System\nCBIT Hacktoberfest 2025 - PS26")
    page = st.radio("Navigation", ["Home","Data","Models","Predict","About"], index=0)
    model_choice = st.selectbox("Select model", ["Logistic Regression","Decision Tree","Random Forest"], index=2)
    viz_type = st.selectbox("Visualization", ["Accuracy Bar","Confusion Matrix","Feature Importance","Class Distribution"])    
    dark = st.checkbox("Dark Theme", value=True)

if dark:
    st.markdown('<div class="dark-bg">', unsafe_allow_html=True)

TEAM = ["Member 1 – Data Engineer","Member 2 – ML Engineer","Member 3 – Analyst","Member 4 – Presenter"]

# Data helpers
DATA_CANDIDATES = [Path("data/dataset.csv"), Path("dataset.csv")]

def get_dataset_path() -> Path|None:
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    return None

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_pickle(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

# Pipeline modules (optional integration)
try:
    from src.data_preprocessing import preprocess_pipeline, select_features
    from src.model_training import split_data, scale_features
    from src.model_evaluation import evaluate_model
except Exception:
    preprocess_pipeline = None
    select_features = None
    split_data = None
    scale_features = None
    evaluate_model = None

# Common
def compute_stats(df: pd.DataFrame):
    total = len(df)
    haz = int(df["Hazardous"].astype(str).isin(["True","true","1",1,True]).sum())
    nonhaz = total - haz
    features = [
        'Relative Velocity km per hr','Miles per hour','Miss Dist.(Astronomical)',
        'Miss Dist.(lunar)','Miss Dist.(kilometers)','Semi Major Axis','Aphelion Dist',
        'Mean Motion','Orbital Period']
    used = [c for c in features if c in df.columns]
    return total,haz,nonhaz,used

# HOME
def render_home(df: pd.DataFrame):
    st.title("🌌 Asteroid Hazard Classification System")
    st.subheader("CBIT Hacktoberfest 2025 - Problem Statement PS26")
    st.write("""Build a binary classifier to predict whether an asteroid is hazardous using velocity, distance, and orbital parameters.""")
    st.markdown("**Team:** "+", ".join(TEAM))
    total,haz,nonhaz,used = compute_stats(df)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown('<div class="metric-card"><div class="metric-title">Total Asteroids</div><div class="metric-value">'+str(total)+'</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card"><div class="metric-title">Hazardous</div><div class="metric-value red">'+str(haz)+'</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-card"><div class="metric-title">Non-Hazardous</div><div class="metric-value green">'+str(nonhaz)+'</div></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-card"><div class="metric-title">Features Used</div><div class="metric-value teal">'+str(len(used))+'</div></div>', unsafe_allow_html=True)
    st.markdown("### 🔍 Objectives")
    st.markdown("- ✅ Train LR/DT/RF models\n- 📊 Compare metrics\n- 🧭 Explain predictions\n- 🖥️ Demo-ready dashboard")

    # Model accuracy bar chart (if models present)
    metrics_df = get_model_metrics(df)
    if metrics_df is not None:
        fig = px.bar(metrics_df, x="Model", y="accuracy", text="accuracy", color="Model", title="Model Accuracy", range_y=[0,1])
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train models via the Models page or run main.py.")

# DATA
def render_data(df: pd.DataFrame):
    st.header("📁 Data Exploration")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("### Descriptive Statistics")
    st.dataframe(df.describe(include='all'), use_container_width=True)

    st.markdown("### Missing Values Heatmap")
    miss = df.isna().astype(int)
    figm = go.Figure(data=go.Heatmap(z=miss.values, x=list(miss.columns), y=list(range(len(miss))), colorscale='Viridis'))
    figm.update_layout(height=400)
    st.plotly_chart(figm, use_container_width=True)

    st.markdown("### Class Distribution")
    cls = df['Hazardous'].astype(int)
    cls_counts = cls.value_counts().rename({0:'Non-Hazardous',1:'Hazardous'})
    figc = px.pie(values=cls_counts.values, names=cls_counts.index, hole=.5, color=cls_counts.index,
                  color_discrete_map={'Hazardous':DANGER,'Non-Hazardous':SUCCESS}, title='Class Distribution')
    st.plotly_chart(figc, use_container_width=True)

    st.markdown("### Feature Distribution")
    num_cols = df.select_dtypes(include=[np.number]).columns
    feat = st.selectbox("Select feature", options=list(num_cols))
    if feat:
        st.plotly_chart(px.histogram(df, x=feat, nbins=40, color='Hazardous'), use_container_width=True)

    st.markdown("### Correlation Heatmap")
    corr = df.select_dtypes(include=[np.number]).corr()
    figcorr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='RdBu'))
    st.plotly_chart(figcorr, use_container_width=True)

# MODELS
def get_model_metrics(df: pd.DataFrame):
    try:
        if preprocess_pipeline and split_data and scale_features and evaluate_model:
            X,y,_ = preprocess_pipeline(str(get_dataset_path()))
            X_tr,X_te,y_tr,y_te = split_data(X.values,y.values)
            X_trs,X_tes,scaler = scale_features(X_tr,X_te)
            metrics = []
            for name,fpath in {
                'Logistic Regression':Path('models/logistic_regression.pkl'),
                'Decision Tree':Path('models/decision_tree.pkl'),
                'Random Forest':Path('models/random_forest.pkl'),
            }.items():
                obj = load_pickle(fpath)
                if obj:
                    m = obj['model']
                    metrics.append({**evaluate_model(m,X_tes,y_te,name), 'Model':name})
            if metrics:
                return pd.DataFrame(metrics)
    except Exception:
        pass
    return None

def render_models(df: pd.DataFrame):
    st.header("🤖 Model Training & Results")
    log_holder = st.expander("Live Logs", expanded=False)
    colA,colB = st.columns([1,2])
    with colA:
        if st.button("Run Training Pipeline", help="Executes main.py and updates models/results"):
            with st.spinner("Running pipeline..."):
                proc = subprocess.Popen([sys.executable, "main.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                logs = io.StringIO()
                for line in iter(proc.stdout.readline, ''):
                    logs.write(line)
                    log_holder.code(logs.getvalue())
                proc.wait()
                st.success("Pipeline complete")
    metrics_df = get_model_metrics(df)
    if metrics_df is not None and not metrics_df.empty:
        st.markdown("### Model Comparison")
        st.dataframe(metrics_df.style.background_gradient(cmap="Blues"), use_container_width=True)
        fig = px.bar(metrics_df, x="Model", y="accuracy", color="Model", title="Accuracy")
        st.plotly_chart(fig, use_container_width=True)
        best_row = metrics_df.sort_values("accuracy", ascending=False).iloc[0]
        st.success(f"🏆 Best Model: {best_row['Model']} (Acc {best_row['accuracy']:.2f})")
        # Confusion matrix (recompute using best model)
        # Use RF feature importance if available
        rf_obj = load_pickle(Path('models/random_forest.pkl'))
        if rf_obj:
            m = rf_obj['model']
            X,y,feat_names = preprocess_pipeline(str(get_dataset_path())) if preprocess_pipeline else (df.drop(columns=['Hazardous']),df['Hazardous'],list(df.columns))
            X_tr,X_te,y_tr,y_te = split_data(X.values,y.values) if split_data else (X.values[:int(.8*len(X))],X.values[int(.8*len(X)):],y.values[:int(.8*len(y))],y.values[int(.8*len(y)):])
            X_trs,X_tes,_ = scale_features(X_tr,X_te) if scale_features else (X_tr,X_te,None)
            y_pred = m.predict(X_tes)
            # plotly confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_te,y_pred)
            figcm = go.Figure(data=go.Heatmap(z=cm, x=['Pred 0','Pred 1'], y=['True 0','True 1'], colorscale='Blues'))
            figcm.update_layout(title='Confusion Matrix')
            st.plotly_chart(figcm, use_container_width=True)
            if hasattr(m,'feature_importances_'):
                imp = pd.Series(m.feature_importances_, index=feat_names[:len(m.feature_importances_)])
                figimp = px.bar(imp.sort_values(ascending=True), orientation='h', title='Feature Importance')
                st.plotly_chart(figimp, use_container_width=True)
        # Downloads
        dlc1,dlc2,dlc3 = st.columns(3)
        with dlc1:
            for f in ['best_model.pkl','logistic_regression.pkl','decision_tree.pkl','random_forest.pkl']:
                p = Path('models')/f
                if p.exists():
                    st.download_button(f"Download {f}", data=open(p,'rb').read(), file_name=f)
        with dlc2:
            rep = Path('results/classification_report.txt')
            if rep.exists():
                st.download_button("Download classification_report.txt", data=open(rep,'rb').read(), file_name='classification_report.txt')
        with dlc3:
            cm = Path('results/confusion_matrix.png')
            if cm.exists():
                st.download_button("Download confusion_matrix.png", data=open(cm,'rb').read(), file_name='confusion_matrix.png')
    else:
        st.warning("No trained models found. Run pipeline to generate models.")

# PREDICT
def render_predict(df: pd.DataFrame):
    st.header("🔮 Live Prediction")
    best = load_pickle(Path('models/best_model.pkl')) or load_pickle(Path('models/random_forest.pkl'))
    if not best:
        st.error("Best model not found. Train models first.")
        return
    model, scaler = best['model'], best['scaler']
    features = [
        'Relative Velocity km per hr','Miles per hour','Miss Dist.(Astronomical)',
        'Miss Dist.(lunar)','Miss Dist.(kilometers)','Semi Major Axis','Aphelion Dist',
        'Mean Motion','Orbital Period']
    avail = [f for f in features if f in df.columns]
    stats = df[avail].describe()
    cols = st.columns(3)
    inputs = []
    for i,f in enumerate(avail):
        m = float(stats.loc['mean',f]) if f in stats.columns else 0.0
        mn = float(stats.loc['min',f]) if f in stats.columns else 0.0
        mx = float(stats.loc['max',f]) if f in stats.columns else m*2+1
        with cols[i%3]:
            inputs.append(st.slider(f, mn, mx, m))
    if st.button("Predict", type="primary"):
        X = np.array(inputs, dtype=float).reshape(1,-1)
        Xs = scaler.transform(X) if scaler else X
        proba = getattr(model,'predict_proba',None)
        if proba:
            p = proba(Xs)[0]
            pred = int(np.argmax(p))
            conf = float(np.max(p))
        else:
            pred = int(model.predict(Xs)[0]); conf = 1.0
            p = [1-conf, conf]
        if pred==1:
            st.markdown(f"<div class='metric-card' style='background:{DANGER}30'><span class='red'>⚠️ HAZARDOUS</span> — Confidence {conf:.2f}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='metric-card' style='background:{SUCCESS}30'><span class='green'>✅ SAFE</span> — Confidence {conf:.2f}</div>", unsafe_allow_html=True)
        st.write({"Safe %":f"{p[0]*100:.1f}%","Hazardous %":f"{p[1]*100:.1f}%"})
        if hasattr(model,'feature_importances_'):
            imp = pd.Series(model.feature_importances_, index=avail[:len(model.feature_importances_)])
            st.plotly_chart(px.bar(imp.sort_values(ascending=True), orientation='h', title='Feature Contribution'), use_container_width=True)

# ABOUT
def render_about():
    st.header("ℹ️ About")
    st.write("Team members:")
    for t in TEAM:
        st.markdown(f"- {t}")
    st.write("Technologies: Streamlit, Plotly, scikit-learn, pandas, numpy")
    st.write("GitHub: https://github.com/your-repo")
    st.write("Acknowledgments: CBIT Hacktoberfest 2025")

# Load data
path = get_dataset_path()
if not path:
    st.error("Dataset not found. Place dataset.csv in data/ or root.")
    st.stop()

df = load_data(path)

# Router
if page=="Home":
    render_home(df)
elif page=="Data":
    render_data(df)
elif page=="Models":
    render_models(df)
elif page=="Predict":
    render_predict(df)
else:
    render_about()

if dark:
    st.markdown('</div>', unsafe_allow_html=True)