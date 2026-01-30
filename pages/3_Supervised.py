import streamlit as st
import pandas as pd
import numpy as np
import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier, XGBRegressor = None, None
import plotly.express as px
import pickle
import base64

# Initialize
st.set_page_config(page_title="Supervised ML - Visio AI", page_icon="🤖", layout="wide")
utils.init_session_state()
utils.load_css()
utils.sidebar_nav()

st.markdown("## 🤖 Supervised Learning")

if st.session_state.updated_df is None:
    st.warning("Please upload a dataset in 'Data Loader' first.")
    if st.button("Go to Data Loader"):
        st.switch_page("pages/1_Data_Loader.py")
    st.stop()

df = st.session_state.updated_df

# Layout
tabs = st.tabs(["🏗️ Build Model", "🔮 Prediction Lab"])

with tabs[0]:
    col_setup, col_train = st.columns([1, 2])
    
    with col_setup:
        st.markdown("### 1. Setup")
        
        target = st.selectbox("Target Variable (Y)", df.columns, key="target_selector")
        st.session_state.target_column = target
        
        # Problem inference
        default_type = "Regression" if (df[target].dtype in ['int64', 'float64'] and df[target].nunique() > 10) else "Classification"
        prob_type = st.radio("Problem Type", ["Classification", "Regression"], index=0 if default_type=="Classification" else 1)
        st.session_state.problem_type = prob_type.lower()
        
        features = st.multiselect("Features (X)", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target])
        st.session_state.feature_columns = features
        
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        random_state = st.number_input("Random Seed", value=42)
        
        if st.button("Prepare Data"):
            if not features:
                st.error("Select features!")
            else:
                X = df[features].copy()
                y = df[target].copy()
                
                # Record feature types for inference later
                st.session_state.feature_types = X.dtypes.to_dict()
                
                # Preprocessing
                for col in X.select_dtypes(include=['object', 'bool']).columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    # Store encoders if we wanted robust inference (simplified here)
                
                if prob_type == "Classification" and y.dtype == 'object':
                    le_y = LabelEncoder()
                    y = le_y.fit_transform(y)
                    st.session_state.target_encoder = le_y
                    
                # Store Scaler
                num_cols = X.select_dtypes(include=['number']).columns
                scaler = StandardScaler()
                if not num_cols.empty:
                    X[num_cols] = scaler.fit_transform(X[num_cols])
                    st.session_state.scaler = scaler
                    st.session_state.num_cols = num_cols
                    
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                st.session_state.X_train = X_train; st.session_state.X_test = X_test
                st.session_state.y_train = y_train; st.session_state.y_test = y_test
                
                st.success(f"Data Prepared. Train: {len(X_train)}")

    with col_train:
        st.markdown("### 2. Training")
        if st.session_state.X_train is not None:
            if st.session_state.problem_type == "classification":
                algos = {
                    "Logistic Regression": LogisticRegression(),
                    "Random Forest": RandomForestClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "SVC": SVC(),
                    "KNN": KNeighborsClassifier(),
                    "Naive Bayes": GaussianNB()
                }
                if XGBClassifier: algos["XGBoost (Gradient Boosting)"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            else:
                algos = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "SVR": SVR(),
                    "KNN": KNeighborsRegressor()
                }
                if XGBRegressor: algos["XGBoost (Gradient Boosting)"] = XGBRegressor()
                
            algo_name = st.selectbox("Algorithm", list(algos.keys()))
            model = algos[algo_name]
            
            if st.button("🚀 Train Model"):
                with st.spinner("Training..."):
                    model.fit(st.session_state.X_train, st.session_state.y_train)
                    st.session_state.trained_model = model
                    st.session_state.selected_algo_name = algo_name
                    
                    y_pred = model.predict(st.session_state.X_test)
                    
                    # Metrics & Plots
                    metrics = {}
                    if st.session_state.problem_type == "classification":
                        metrics["Accuracy"] = accuracy_score(st.session_state.y_test, y_pred)
                        metrics["F1"] = f1_score(st.session_state.y_test, y_pred, average='weighted', zero_division=0)
                        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                        
                        fig = px.imshow(confusion_matrix(st.session_state.y_test, y_pred), text_auto=True, title="Confusion Matrix")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        metrics["MSE"] = mean_squared_error(st.session_state.y_test, y_pred)
                        metrics["R2"] = r2_score(st.session_state.y_test, y_pred)
                        st.metric("R2 Score", f"{metrics['R2']:.4f}")
                        
                        fig = px.scatter(x=st.session_state.y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    st.session_state.model_metrics = metrics
                    
                    # Download
                    model_pkl = pickle.dumps(model)
                    b64 = base64.b64encode(model_pkl).decode()
                    st.download_button("📥 Download PKL", data=base64.b64decode(b64), file_name=f"{algo_name}.pkl")

with tabs[1]:
    st.markdown("### 🔮 Prediction Lab")
    st.markdown("Test your trained model with manual inputs.")
    
    if st.session_state.trained_model is not None and st.session_state.feature_columns:
        st.info(f"Model: **{st.session_state.selected_algo_name}**")
        
        input_data = {}
        # Dynamic inputs based on feature columns
        for col in st.session_state.feature_columns:
             # Basic handling: Text inputs for everything, could be improved with type checking from st.session_state.feature_types
             dtype = str(st.session_state.feature_types.get(col, ""))
             
             if "float" in dtype or "int" in dtype:
                 input_data[col] = st.number_input(f"{col}", value=0.0)
             else:
                 input_data[col] = st.text_input(f"{col} (String)")
        
        if st.button("Predict Outcome"):
            try:
                # Prepare input DF
                input_df = pd.DataFrame([input_data])
                
                # Apply same preprocessing
                # Note: In a real prod app, we'd use saved pipelines. Here we do basic approximation using saved scaler.
                # Re-encoding labels for manual input is tricky without saving the encoders for each column.
                # We'll assume numeric input for simplicity or handle basic numeric scaling.
                
                # 1. Label Encode (Basic support: assumes input is numerical rep or similar)
                for col in input_df.select_dtypes(include=['object']).columns:
                     # Warn user
                     st.warning(f"Column '{col}' expected categorical. Auto-encoding might differ from training. Please input numerical code if possible.")
                     # Simple hash or try to convert float
                     try: input_df[col] = input_df[col].astype(float)
                     except: pass
                
                # 2. Scale
                if hasattr(st.session_state, 'scaler') and hasattr(st.session_state, 'num_cols'):
                    input_df[st.session_state.num_cols] = st.session_state.scaler.transform(input_df[st.session_state.num_cols])
                
                prediction = st.session_state.trained_model.predict(input_df)
                
                # Inverse transform target if classification
                res = prediction[0]
                if hasattr(st.session_state, 'target_encoder'):
                    res = st.session_state.target_encoder.inverse_transform([int(res)])[0]
                    
                st.success(f"### Prediction: {res}")
                
            except Exception as e:
                st.error(f"Prediction failed: {e}. (Ensure inputs match training data format)")
    else:
        st.warning("Please train a model in the 'Build Model' tab first.")
