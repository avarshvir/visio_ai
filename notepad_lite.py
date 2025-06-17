import streamlit as st
from datetime import datetime

def render_notepad():
    """
    Renders a simple, session-based notepad page.
    """
    st.markdown("<h2 style='text-align: center; color: #4A90E2;'>📝 Note -- Lite</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Jot down your thoughts, findings, or reminders. Your notes are saved for the current session.</p>", unsafe_allow_html=True)

    # Initialize the note content in session state if it doesn't exist
    if 'notepad_text' not in st.session_state:
        st.session_state.notepad_text = "## My Analysis Notes\n\n- Finding 1:\n- Finding 2:\n"

    # --- DEFINE THE CALLBACK FUNCTION ---
    # This function will be called when the button is clicked.
    # It modifies the session state *before* the page is re-rendered.
    def clear_note_callback():
        st.session_state.notepad_text = ""
    # ------------------------------------

    st.text_area(
        "Your Notes",
        key='notepad_text',
        height=400,
        help="Your text is saved automatically as you type."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # --- ATTACH THE CALLBACK TO THE BUTTON ---
        # Instead of an if-block, we use the on_click parameter.
        st.button(
            "🗑️ Clear Note",
            on_click=clear_note_callback,
            help="Click to permanently delete the text in the notepad."
        )
        # -----------------------------------------

    with col2:
        st.download_button(
            label="📥 Download Note as .txt",
            data=st.session_state.notepad_text,
            file_name=f"visio_ai_note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )