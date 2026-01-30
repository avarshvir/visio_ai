import streamlit as st
import pandas as pd
import numpy as np
import utils
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Initialize
st.set_page_config(page_title="Unsupervised ML - Visio AI", page_icon="🔮", layout="wide")
utils.init_session_state()
utils.load_css()
utils.sidebar_nav()

st.markdown("## 🔮 Unsupervised Learning")

if st.session_state.updated_df is None:
    st.warning("Please upload a dataset in 'Data Loader' first.")
    if st.button("Go to Data Loader"):
        st.switch_page("pages/1_Data_Loader.py")
    st.stop()
    
df = st.session_state.updated_df
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found for unsupervised learning!")
    st.stop()

tab1, tab2 = st.tabs(["🧩 Clustering (K-Means)", "📉 Dimensionality Reduction (PCA)"])

with tab1:
    st.markdown("### K-Means Clustering")
    
    col_setup, col_viz = st.columns([1, 2])
    
    with col_setup:
        features = st.multiselect("Select Features", numeric_cols, default=numeric_cols[:2])
        n_clusters = st.slider("Number of Clusters (K)", 2, 10, 3)
        
        if st.button("Run K-Means"):
            if len(features) < 2:
                st.error("Select at least 2 features.")
            else:
                X = df[features].dropna()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(X_scaled)
                
                # Store result
                X['Cluster'] = clusters.astype(str)
                st.session_state.unsupervised_results = X
                st.success("Clustering Complete!")
                
    with col_viz:
        if 'unsupervised_results' in st.session_state:
            res_df = st.session_state.unsupervised_results
            features_used = [c for c in res_df.columns if c != 'Cluster']
            
            if len(features_used) >= 3:
                st.info("Visualizing first 3 features in 3D")
                fig = px.scatter_3d(res_df, x=features_used[0], y=features_used[1], z=features_used[2], 
                                    color='Cluster', title=f"K-Means (K={n_clusters})")
            else:
                fig = px.scatter(res_df, x=features_used[0], y=features_used[1], 
                                 color='Cluster', title=f"K-Means (K={n_clusters})")
            
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### PCA - Principal Component Analysis")
    
    col_pca_setup, col_pca_viz = st.columns([1, 2])
    
    with col_pca_setup:
        pca_features = st.multiselect("Features for PCA", numeric_cols, default=numeric_cols, key="pca_feat")
        n_components = st.slider("Components", 2, min(len(pca_features), 10), 2)
        
        if st.button("Run PCA"):
             if len(pca_features) < 2:
                st.error("Select at least 2 features.")
             else:
                X_pca = df[pca_features].dropna()
                scaler_pca = StandardScaler()
                X_pca_scaled = scaler_pca.fit_transform(X_pca)
                
                pca = PCA(n_components=n_components)
                components = pca.fit_transform(X_pca_scaled)
                
                pca_df = pd.DataFrame(data=components, columns=[f"PC{i+1}" for i in range(n_components)])
                st.session_state.pca_result = pca_df
                st.session_state.pca_variance = pca.explained_variance_ratio_
                st.success("PCA Complete!")

    with col_pca_viz:
        if 'pca_result' in st.session_state and st.session_state.pca_result is not None:
            res_pca = st.session_state.pca_result
            
            # Variance Plot
            var_df = pd.DataFrame({'Component': [f"PC{i+1}" for i in range(len(st.session_state.pca_variance))],
                                   'Variance Explained': st.session_state.pca_variance})
            fig_var = px.bar(var_df, x='Component', y='Variance Explained', title="Explained Variance Ratio")
            st.plotly_chart(fig_var, use_container_width=True)
            
            # Scatter Plot
            if n_components >= 3:
                 fig_pca = px.scatter_3d(res_pca, x='PC1', y='PC2', z='PC3', title="PCA 3D Projection")
            else:
                 fig_pca = px.scatter(res_pca, x='PC1', y='PC2', title="PCA 2D Projection")
            
            st.plotly_chart(fig_pca, use_container_width=True)
