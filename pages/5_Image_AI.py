import streamlit as st
import base64
import requests
from PIL import Image
import io
import utils

# Initialize
st.set_page_config(page_title="Image AI - Visio AI", page_icon="👁️", layout="wide")
utils.init_session_state()
utils.load_css()
utils.sidebar_nav()

# Initialize specific session state for this page if needed (though utils handles defaults)
if "viz_ai_img_results" not in st.session_state:
    st.session_state.viz_ai_img_results = []
if "viz_ai_img_current_index" not in st.session_state:
    st.session_state.viz_ai_img_current_index = -1

st.markdown("## 👁️ Image Analysis AI")
st.markdown("Upload an image and ask the AI anything about it.")

# ... (Include the rest of the original logic but with cleaned up UI)
# Using the core logic from the original file but adapted to the new theme

# === Results Manager ===
if st.session_state.viz_ai_img_results:
    st.subheader("📊 Analysis History")
    cols = st.columns(min(4, len(st.session_state.viz_ai_img_results)))
    for i, result in enumerate(st.session_state.viz_ai_img_results):
        with cols[i]:
            st.image(result["image"], caption=f"#{len(st.session_state.viz_ai_img_results)-i}", use_container_width=True)
            if st.button(f"View #{len(st.session_state.viz_ai_img_results)-i}", key=f"view_{i}"):
                st.session_state.viz_ai_img_current_index = i
    st.markdown("---")

# === Current Result ===
if st.session_state.viz_ai_img_current_index >= 0:
    data = st.session_state.viz_ai_img_results[st.session_state.viz_ai_img_current_index]
    with st.expander(f"Clipboard", expanded=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(data["image"])
        with c2:
            st.markdown(f"**Model:** {data['model']}")
            st.markdown(f"**Prompt:** {data['prompt']}")
            st.info(data['analysis'])
            
    if st.button("Close View"):
        st.session_state.viz_ai_img_current_index = -1
        st.rerun()

# === New Analysis ===
st.markdown("### ➕ New Analysis")
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Preview", use_container_width=True)

with col2:
    prompt_text = st.text_area("Prompt", "Describe this image in detail.", height=150)
    model_selection = st.selectbox("Model", [
        "nvidia/nemotron-nano-12b-v2-vl:free"
    ])
    
    if st.button("🚀 Analyze", type="primary"):
        if not uploaded_image:
            st.error("Upload image first.")
        else:
             with st.spinner("Analyzing..."):
                try:
                    # Mocking API call logic wrapper or re-implementing it here
                    # For production, this should ideally be in utils.py or kept here if specific
                    
                    api_key = st.secrets.get("OPENROUTER_API_KEY", "")
                    if not api_key:
                        st.warning("OPENROUTER_API_KEY not found in secrets. Using dummy response for demo.")
                        ai_response = "This is a simulated response because API key is missing. The image looks amazing!"
                    else:
                        buffered = io.BytesIO()
                        image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        
                        response = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                            json={
                                "model": model_selection,
                                "messages": [{"role": "user", "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                                    {"type": "text", "text": prompt_text}
                                ]}]
                            }
                        )
                        if response.status_code == 200:
                            ai_response = response.json()["choices"][0]["message"]["content"]
                        else:
                            ai_response = f"Error: {response.text}"

                    # Save
                    st.session_state.viz_ai_img_results.insert(0, {
                        "image": image, "prompt": prompt_text, "analysis": ai_response, "model": model_selection
                    })
                    st.session_state.viz_ai_img_current_index = 0
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {e}")
