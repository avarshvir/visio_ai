import streamlit as st
import pandas as pd
import numpy as np

def init_session_state():
    """Initialize all session state variables used across the app."""
    defaults = {
        'updated_df': None,
        'original_df_uploaded': False,
        'last_uploaded_file_name': None,
        # Legacy Feature States
        'notepad_text': "## Annual Review Notes\n\n- Key Metric 1:\n- Action Item:",
        'calc_display': '0',
        'word_cloud_result': None,
        # ML States
        'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
        'trained_model': None, 'model_metrics': None,
        'viz_ai_img_results': [], 'viz_ai_img_current_index': -1
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def load_css(file_path="styles.css"):
    """Load global CSS."""
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def sidebar_nav():
    """Render the sidebar navigation links."""
    st.sidebar.markdown("### Resources")
    
    links = {
        "📖 User Guide": "https://jaiho-labs.onrender.com/pages/products_resources/docs/visio_ai_docs/visio_helper.html",
        "📄 Documentation": "https://jaiho-labs.onrender.com/pages/products_resources/docs/visio_ai_docs/visio_docs.html",
        "ℹ️ About": "https://jaiho-labs.onrender.com/pages/products_resources/docs/visio_ai_docs/visio_about.html",
        "⚡ Elite Access": "https://jaiho-labs.onrender.com/pages/products_resources/docs/visio_ai_docs/get_elite_access.html"
    }
    
    for label, url in links.items():
        st.sidebar.markdown(f'<a href="{url}" target="_blank" style="text-decoration: none; color: #64748B; display: block; padding: 5px 0;">{label}</a>', unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size: 0.75rem; color: #94A3B8;">
            © 2026 Jaiho Labs<br>
            Visio AI Platform v2.0
        </div>
        """,
        unsafe_allow_html=True
    )

# Alias for backward compatibility with pages not yet updated
sidebar_credits = sidebar_nav

def safe_read_csv(file):
    """Safely read CSV/Excel/Text files."""
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file)
        elif file.name.endswith(".txt"):
            return pd.read_csv(file, delimiter="\t")
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None
