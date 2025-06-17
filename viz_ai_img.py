import streamlit as st
import base64
import requests
from PIL import Image
import io

def analyze_image_ui():
    """
    Renders the UI for the Viz AI Image Analysis tool and handles the
    logic for sending requests to the OpenRouter API.
    """
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #4A90E2;'>🤖 Viz AI (Image)</h2>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: grey;'>Uncover hidden patterns and details in your images.</h5>", unsafe_allow_html=True)

    # Use a two-column layout
    col1, col2 = st.columns(2)

    with col1:
        uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        
        if uploaded_image:
            # Display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
    with col2:
        prompt_text = st.text_area(
            "Your Prompt:",
            "Describe this image in detail. What are the key objects, arrangements, and potential hidden patterns or meanings?",
            height=150
        )

        
        # Add a select box for the model
        model_selection = st.selectbox(
            "Choose a model:",
            (
                "meta-llama/llama-4-maverick:free",
                "opengvlab/internvl3-14b:free",
                "mistralai/mistral-small-3.1-24b-instruct:free",
                "google/gemma-3-27b-it:free",
            )
        )

        analyze_button = st.button("Analyze Image ✨")

    if analyze_button and uploaded_image:
        if not prompt_text.strip():
            st.error("Please enter a prompt.")
            return

        with st.spinner(f"AI is analyzing the image using {model_selection}..."):
            try:
                # Get the API key from secrets
                api_key = st.secrets["OPENROUTER_API_KEY"]
                if not api_key:
                    st.error("OpenRouter API key is not set. Please add it to your secrets.")
                    return

                # Convert image to base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model_selection,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                                    {"type": "text", "text": prompt_text}
                                ]
                            }
                        ]
                    }
                )

                response.raise_for_status()  # Will raise an HTTPError for bad responses (4xx or 5xx)
                
                result = response.json()
                ai_response = result['choices'][0]['message']['content']
                
                st.markdown("---")
                st.subheader("Analysis Result:")
                st.markdown(ai_response)
                # --- ADD THIS BLOCK ---
                # Save results to session state for the report
                st.session_state['viz_ai_img_result'] = {
                    "image": image,  # The PIL Image object
                    "prompt": prompt_text,
                    "analysis": ai_response,
                    "model": model_selection
                }
                st.success("✅ Analysis saved to the session report.")

            except requests.exceptions.HTTPError as http_err:
                st.error(f"HTTP error occurred: {http_err} - {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
    elif analyze_button and not uploaded_image:
        st.warning("Please upload an image first.")