import streamlit as st
from datetime import datetime
import utils

st.set_page_config(page_title="Notes - Visio AI", page_icon="📝", layout="wide")
utils.init_session_state()
utils.load_css()
utils.sidebar_credits()

# Make sure notes persist when switching pages
if "notes" not in st.session_state:
    st.session_state.notes = ""

st.markdown("## 📝 Visio Notes")
st.markdown("Your notes are saved automatically and will stay here even if you switch to other pages.")

# Big text area for notes
user_notes = st.text_area(
    "Write your notes here...",
    value=st.session_state.notes,
    height=500,
    label_visibility="collapsed"
)

# Save notes to session state every time user types
st.session_state.notes = user_notes

col1, col2 = st.columns([1, 4])
with col1:
    if st.session_state.notes:
        st.download_button(
            label="💾 Download Notes",
            data=st.session_state.notes,
            file_name=f"my_notes_{datetime.now().strftime('%Y-%m-%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )

st.caption("Notes are persistent across your session.")
