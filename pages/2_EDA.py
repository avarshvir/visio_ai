import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
import utils

# Initialize
st.set_page_config(page_title="EDA - Visio AI", page_icon="📊", layout="wide")
utils.init_session_state()
utils.load_css()
utils.sidebar_nav()

st.markdown("## 📊 Exploratory Data Analysis")

if st.session_state.updated_df is None:
    st.warning("Please upload a dataset in 'Data Loader' first.")
    if st.button("Go to Data Loader"):
        st.switch_page("pages/1_Data_Loader.py")
    st.stop()

df = st.session_state.updated_df
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Tabs for organization
tab1, tab2, tab3 = st.tabs(["📈 Visualization", "📋 Statistics", "🔍 Advanced"])

with tab1:
    col_ctrl, col_plot = st.columns([1, 3])
    
    with col_ctrl:
        st.markdown("### Settings")
        
        # --- Smart Suggestions ---
        suggested_plot = "Scatter Plot" # Default
        if len(numerical_cols) >= 2:
            suggested_plot = "Scatter Plot"
        elif len(numerical_cols) == 1 and len(categorical_cols) >= 1:
            suggested_plot = "Box Plot"
        elif len(numerical_cols) == 1:
            suggested_plot = "Histogram"
            
        st.info(f"💡 Suggestion: **{suggested_plot}**")
        
        plot_library = st.radio("Library", ["Interactive (Plotly)", "Static (Seaborn)"], horizontal=True)
        
        plot_type = st.selectbox("Chart Type", [
            "Scatter Plot", "Bar Chart", "Histogram", "Box Plot", 
            "Heatmap", "Line Chart", "Violin Plot", "3D Scatter", "Pair Plot"
        ])
        
        x_col = st.selectbox("X Axis", df.columns)
        y_col = st.selectbox("Y Axis (Optional)", ["None"] + list(df.columns))
        color_col = st.selectbox("Color (Optional)", ["None"] + list(df.columns))
        
        btn_plot = st.button("Generate Plot", type="primary")

    with col_plot:
        if btn_plot:
            try:
                if plot_library == "Interactive (Plotly)":
                    fig = None
                    if plot_type == "Scatter Plot":
                        fig = px.scatter(df, x=x_col, y=None if y_col=="None" else y_col, 
                                         color=None if color_col=="None" else color_col, title=f"{x_col} vs {y_col}")
                    elif plot_type == "Bar Chart":
                        fig = px.bar(df, x=x_col, y=None if y_col=="None" else y_col,
                                     color=None if color_col=="None" else color_col)
                    elif plot_type == "Histogram":
                        fig = px.histogram(df, x=x_col, color=None if color_col=="None" else color_col)
                    elif plot_type == "Box Plot":
                        fig = px.box(df, x=x_col, y=None if y_col=="None" else y_col,
                                     color=None if color_col=="None" else color_col)
                    elif plot_type == "Heatmap":
                        if len(numerical_cols) > 1:
                            fig = px.imshow(df[numerical_cols].corr(), text_auto=True, title="Correlation Heatmap")
                        else:
                            st.warning("Need at least 2 numerical columns for a heatmap.")
                    elif plot_type == "3D Scatter":
                        if len(numerical_cols) >= 3:
                            z_col = st.selectbox("Z Axis (for 3D)", numerical_cols, key="z_3d")
                            fig = px.scatter_3d(df, x=x_col, y=None if y_col=="None" else y_col, z=z_col,
                                                color=None if color_col=="None" else color_col)
                        else:
                             st.warning("Need at least 3 numerical columns.")
                    elif plot_type == "Violin Plot":
                         fig = px.violin(df, x=x_col, y=None if y_col=="None" else y_col, color=None if color_col=="None" else color_col)
                    elif plot_type == "Line Chart":
                        fig = px.line(df, x=x_col, y=None if y_col=="None" else y_col, color=None if color_col=="None" else color_col)
                    else:
                        st.info("This plot type is better suited for Static mode or custom builder.")

                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                else: # Static (Seaborn/Matplotlib)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if plot_type == "Scatter Plot":
                        sns.scatterplot(data=df, x=x_col, y=None if y_col=="None" else y_col, 
                                        hue=None if color_col=="None" else color_col, ax=ax)
                    elif plot_type == "Bar Chart":
                        sns.barplot(data=df, x=x_col, y=None if y_col=="None" else y_col, 
                                    hue=None if color_col=="None" else color_col, ax=ax)
                    elif plot_type == "Histogram":
                        sns.histplot(data=df, x=x_col, hue=None if color_col=="None" else color_col, kde=True, ax=ax)
                    elif plot_type == "Box Plot":
                        sns.boxplot(data=df, x=x_col, y=None if y_col=="None" else y_col, 
                                    hue=None if color_col=="None" else color_col, ax=ax)
                    elif plot_type == "Heatmap":
                         if len(numerical_cols) > 1:
                            sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
                    elif plot_type == "Pair Plot":
                        st.info("Pair Plot generates a grid. It might take a moment.")
                        fig = sns.pairplot(df[numerical_cols])
                        # PairGrid is a different object than AxesSubplot, handling differently
                        st.pyplot(fig.fig) # Access the figure from PairGrid
                        fig = None # Prevent re-plotting below
                    
                    if fig:
                        st.pyplot(fig)

            except Exception as e:
                st.error(f"Error generating plot: {e}")

with tab2:
    st.markdown("### Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Skewness")
        st.dataframe(df.skew(numeric_only=True), use_container_width=True)
    with col2:
        st.markdown("#### Kurtosis")
        st.dataframe(df.kurtosis(numeric_only=True), use_container_width=True)

with tab3:
    st.markdown("### Profile Report")
    st.info("Generating a pandas profiling report can be resource intensive.")
    if st.button("Generate Basic Profile"):
        # Fix: Use StringIO buffer instead of list
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
        st.markdown("#### Unique Values")
        st.write(df.nunique())
        
        st.markdown("#### Missing Values")
        st.write(df.isnull().sum())
