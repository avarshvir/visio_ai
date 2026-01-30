import streamlit as st
from datetime import datetime
import utils

st.set_page_config(page_title="Notes - Visio AI", page_icon="📝", layout="wide")
utils.init_session_state()
utils.load_css()
utils.sidebar_nav()

st.markdown("## 📝 Analyst Notepad")
st.markdown("Scratchpad for your insights. Persists during your session.")

# Initialize the note content in session state if it doesn't exist
if 'notepad_text' not in st.session_state:
    st.session_state.notepad_text = "## Analysis Log\n\n- [ ] Check outliers\n- [ ] Compare Random Forest vs XGBoost\n"

def clear_note():
    st.session_state.notepad_text = ""

# Layout
col_text, col_actions = st.columns([3, 1])

with col_text:
    user_notes = st.text_area(
        "Notes Content",
        value=st.session_state.notepad_text,
        height=500,
        label_visibility="collapsed",
        key="note_input"
    )
    # Update state on every interaction (implicitly handled by key binding in simpler cases, 
    # but explicit assignment ensures sync if complex logic involved)
    st.session_state.notepad_text = user_notes

with col_actions:
    st.markdown("### Actions")
    st.markdown("Manage your current session notes.")
    
    st.download_button(
        label="💾 Download TXT",
        data=st.session_state.notepad_text,
        file_name=f"visio_notes_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    if st.button("🗑️ Clear Notes", type="primary", use_container_width=True):
        clear_note()
        st.rerun()

    st.info("💡 **Tip**: These notes are temporary to your session window. Download them if you need to keep them forever.")
