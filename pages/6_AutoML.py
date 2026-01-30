import streamlit as st
import pandas as pd
import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier, XGBRegressor = None, None

st.set_page_config(page_title="AutoML - Visio AI", page_icon="🏎️", layout="wide")
utils.init_session_state()
utils.load_css()
utils.sidebar_nav()

st.markdown("## 🏎️ AutoML: Model Comparison")

if st.session_state.updated_df is None:
    st.warning("Please upload a dataset in 'Data Loader'.")
    st.stop()
    
df = st.session_state.updated_df

col1, col2 = st.columns([1, 2])

with col1:
    target = st.selectbox("Target Variable", df.columns)
    problem_type = st.radio("Problem Type", ["Classification", "Regression"])
    features = st.multiselect("Features", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target])
    
    if st.button("Run AutoML 🚀"):
        if not features:
            st.error("Select features.")
        else:
            with st.spinner("Running AutoML..."):
                X = df[features].copy()
                y = df[target].copy()
                
                # Preprocessing
                for col in X.select_dtypes(include=['object']).columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                
                if problem_type == "Classification" and y.dtype == 'object':
                    y = LabelEncoder().fit_transform(y)
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                results = []
                
                if problem_type == "Classification":
                    models = {
                        "Logistic Regression": LogisticRegression(),
                        "Random Forest": RandomForestClassifier(),
                        "Decision Tree": DecisionTreeClassifier(),
                        "SVC": SVC()
                    }
                    if XGBClassifier: models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                    metric_name = "Accuracy"
                else:
                    models = {
                        "Linear Regression": LinearRegression(),
                        "Random Forest": RandomForestRegressor(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "SVR": SVR()
                    }
                    if XGBRegressor: models["XGBoost"] = XGBRegressor()
                    metric_name = "R2 Score"
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    if problem_type == "Classification":
                        score = accuracy_score(y_test, y_pred)
                    else:
                        score = r2_score(y_test, y_pred)
                    
                    results.append({"Model": name, metric_name: score})
                
                st.session_state.automl_results = pd.DataFrame(results).sort_values(by=metric_name, ascending=False)
                st.success("AutoML Complete!")

with col2:
    if 'automl_results' in st.session_state:
        st.markdown("### Leaderboard")
        res_df = st.session_state.automl_results
        st.dataframe(res_df, use_container_width=True)
        
        best_model = res_df.iloc[0]
        st.info(f"🏆 Best Model: **{best_model['Model']}** with Score: **{best_model[list(best_model.index)[1]]:.4f}**")
        
        # Simple bar chart
        st.bar_chart(res_df.set_index("Model"))
