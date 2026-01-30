import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from wordcloud import WordCloud, STOPWORDS
import utils

# Optional imports for file types
try:
    import PyPDF2
    from docx import Document
except ImportError:
    pass

st.set_page_config(page_title="Word Cloud - Visio AI", page_icon="☁️", layout="wide")
utils.init_session_state()
utils.load_css()
utils.sidebar_nav()

st.markdown("## ☁️ Word Cloud Generator")

def extract_text(file):
    text = ""
    try:
        if file.name.endswith('.pdf'):
            if 'PyPDF2' not in globals(): return "Error: PyPDF2 not installed."
            read = PyPDF2.PdfReader(file)
            for page in read.pages: text += page.extract_text()
        elif file.name.endswith('.docx'):
            if 'Document' not in globals(): return "Error: python-docx not installed."
            doc = Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif file.name.endswith('.txt'):
            text = file.read().decode('utf-8')
        elif file.name.endswith(('.csv', '.xlsx')):
            df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
            # Combine all string columns
            str_df = df.select_dtypes(include=['object', 'string'])
            for col in str_df.columns:
                text += " ".join(str_df[col].dropna().astype(str))
    except Exception as e:
        return f"Error reading file: {e}"
    return text

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 1. Source")
    upl_file = st.file_uploader("Upload Document", type=['pdf', 'docx', 'txt', 'csv', 'xlsx'])
    
    text_data = ""
    if upl_file:
        text_data = extract_text(upl_file)
        if text_data.startswith("Error"):
            st.error(text_data)
            text_data = ""
        else:
            st.success(f"Loaded {len(text_data)} chars")

    st.markdown("### 2. Settings")
    bg_color = st.color_picker("Background", "#FFFFFF")
    color_map = st.selectbox("Palette", ["viridis", "plasma", "inferno", "magma", "ocean"])
    max_words = st.slider("Max Words", 50, 500, 200)

with col2:
    st.markdown("### 3. Result")
    if text_data:
        if st.button("Generate Cloud", type="primary"):
            wc = WordCloud(
                width=800, height=400,
                background_color=bg_color,
                colormap=color_map,
                max_words=max_words,
                stopwords=STOPWORDS
            ).generate(text_data)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
            # Save
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            st.download_button("📥 Download PNG", buf.getvalue(), "wordcloud.png", "image/png")
    else:
        st.info("Upload a document to generate a cloud.")
