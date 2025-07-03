import streamlit as st
from PIL import Image
from caption_generator import generate_captions_and_hashtags

st.title("📸 AI Instagram Caption Assistant")

option = st.radio("Choose input type:", ["Upload Photo", "Enter Description"])

if option == "Upload Photo":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            if st.button("Generate Captions"):
                try:
                    output = generate_captions_and_hashtags(image=image)
                    st.subheader("Captions")
                    st.markdown(output["Captions"])

                    st.subheader("Hashtags")
                    st.markdown(" ".join(output["Hashtags"]))
                except Exception as e:
                    st.error(f"Error generating captions: {e}")
        except Exception as e:
            st.error(f"Failed to open image: {e}")

elif option == "Enter Description":
    description = st.text_area("Enter a short description of your post:")
    if st.button("Generate Captions"):
        try:
            output = generate_captions_and_hashtags(text=description)
            st.subheader("Captions")
            st.markdown(output["Captions"])

            st.subheader("Hashtags")
            st.markdown(" ".join(output["Hashtags"]))
        except Exception as e:
            st.error(f"Error generating captions: {e}")
