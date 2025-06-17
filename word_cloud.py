import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import io

# Libraries for file processing
import PyPDF2
from docx import Document

def extract_text_from_file(uploaded_file):
    """Extracts text from various file formats."""
    if uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif uploaded_file.name.endswith('.docx'):
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    elif uploaded_file.name.endswith('.txt'):
        # To read bytes, decode it to string
        return uploaded_file.read().decode('utf-8')
    elif uploaded_file.name.endswith(('.csv', '.xlsx')):
        return pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
    return None

def render_word_cloud_page():
    """
    Renders the UI and logic for the Word Cloud Generator page.
    """
    st.markdown("<h2 style='text-align: center; color: #4A90E2;'>😶‍🌫️ Word Cloud Generator</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Create beautiful word clouds from your text data. Supports PDF, DOCX, TXT, CSV, and Excel files.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt', 'csv', 'xlsx']
    )

    text_data = None

    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            extracted_content = extract_text_from_file(uploaded_file)

        if isinstance(extracted_content, pd.DataFrame):
            st.info("CSV/Excel file detected. Please select the column to generate the word cloud from.")
            df = extracted_content
            text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()

            if not text_columns:
                st.error("No text-based columns found in the uploaded file.")
                return

            column_to_use = st.selectbox("Select a column:", text_columns)
            if column_to_use:
                text_data = " ".join(df[column_to_use].dropna().astype(str))
        else:
            text_data = extracted_content

    if text_data:
        st.markdown("---")
        st.subheader("Customize Your Word Cloud")

        col1, col2 = st.columns(2)
        with col1:
            colormap = st.selectbox("Color Scheme", ["viridis", "plasma", "inferno", "magma", "cividis", "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds"])
            max_words = st.slider("Maximum Words", 50, 500, 200)
            bg_color = st.color_picker("Background Color", "#FFFFFF")
            
        with col2:
            contour_width = st.slider("Contour Width", 0.0, 5.0, 0.0, 0.1)
            contour_color = st.color_picker("Contour Color", "#0000FF")
            add_stopwords = st.text_area("Add Custom Stopwords (comma-separated)")

        if st.button("Generate Word Cloud ✨"):
            with st.spinner("Creating your masterpiece..."):
                custom_stopwords = set(STOPWORDS)
                if add_stopwords:
                    custom_stopwords.update(add_stopwords.lower().split(','))

                try:
                    wordcloud = WordCloud(
                        width=1200,
                        height=600,
                        background_color=bg_color,
                        stopwords=custom_stopwords,
                        max_words=max_words,
                        colormap=colormap,
                        contour_width=contour_width,
                        contour_color=contour_color
                    ).generate(text_data)

                    st.markdown("---")
                    st.subheader("Generated Word Cloud")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    # --- ADD THIS BLOCK ---
                    st.session_state['word_cloud_result'] = {
                        "figure": fig, # The matplotlib figure object
                        "source": uploaded_file.name,
                        "settings": f"Colors: {colormap}, Max Words: {max_words}"
                    }
                    st.success("✅ Word cloud saved to the session report.")
                    # ----------------------

                    # Create a download button for the image
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight')
                    st.download_button(
                        label="📥 Download Image",
                        data=buf.getvalue(),
                        file_name="word_cloud.png",
                        mime="image/png"
                    )

                except Exception as e:
                    st.error(f"An error occurred while generating the word cloud: {e}")