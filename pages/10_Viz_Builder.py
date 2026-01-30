import streamlit as st
import plotly.express as px
import utils

st.set_page_config(page_title="Viz Builder - Visio AI", page_icon="🎨", layout="wide")
utils.init_session_state()
utils.load_css()
utils.sidebar_nav()

st.markdown("## 🎨 Visual Builder")

if st.session_state.updated_df is None:
    st.warning("Load data first.")
    st.stop()
    
df = st.session_state.updated_df

col_ctrl, col_viz = st.columns([1, 2])

with col_ctrl:
    st.markdown("### Controls")
    
    plot_type = st.selectbox("Type", ["Scatter", "Bar", "Line", "Histogram", "Box"])
    x_axis = st.selectbox("X Axis", df.columns)
    y_axis = st.selectbox("Y Axis (Optional)", [None] + list(df.columns))
    color = st.selectbox("Color (Optional)", [None] + list(df.columns))
    
    title = st.text_input("Title", "My Custom Plot")
    theme = st.selectbox("Theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"])

with col_viz:
    st.markdown("### Canvas")
    try:
        if plot_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color, title=title, template=theme)
        elif plot_type == "Bar":
            fig = px.bar(df, x=x_axis, y=y_axis, color=color, title=title, template=theme)
        elif plot_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis, color=color, title=title, template=theme)
        elif plot_type == "Histogram":
            fig = px.histogram(df, x=x_axis, color=color, title=title, template=theme)
        elif plot_type == "Box":
            fig = px.box(df, x=x_axis, y=y_axis, color=color, title=title, template=theme)
            
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not render: {e}")
