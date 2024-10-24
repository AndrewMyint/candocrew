import logging
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from utils.text_extraction import extract_text_from_image, extract_transaction_data

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def render():
    # Page Header with Icon
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 10px;'>
            <img src='https://cdn-icons-png.flaticon.com/512/5262/5262072.png' width='40'/>
            <h2>Image to Text Extraction Demo</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Introduction
    st.markdown("""
    This demo showcases the process of extracting text from images using advanced OCR techniques. 
    We'll walk through each step of the process, from preprocessing to final text extraction.
    
    ### How it works:
    1. **Image Preprocessing**: We enhance image quality through multiple steps
    2. **OCR Processing**: Using Tesseract OCR engine for text extraction
    3. **Data Parsing**: Structured information extraction using regex patterns
    """)

    st.markdown("---")

    # File Upload Section with Information
    st.markdown(
        """
        <div style='background-color: rgba(61, 157, 243, 0.1); padding: 20px; border-radius: 10px; border: 1px solid rgba(61, 157, 243, 0.3);'>
            <h4 style='color: var(--text-color, currentColor);'>📁 Upload Your Image</h4>
            <p style='color: var(--text-color, currentColor);'>Support formats: PNG, JPG, or JPEG</p>
            <p style='color: var(--text-color, currentColor);'>For best results, ensure your image:</p>
            <ul style='color: var(--text-color, currentColor);'>
                <li>Has good lighting and contrast</li>
                <li>Text is clearly visible</li>
                <li>Minimal background noise</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an image file",
        accept_multiple_files=False,
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file:
        try:
            # Read the image file
            original_image = Image.open(uploaded_file)
            image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

            # Initialize session state
            if "extracted_text" not in st.session_state:
                st.session_state.extracted_text = None
            if "text_extracted" not in st.session_state:
                st.session_state.text_extracted = False
            if "text_parsed" not in st.session_state:
                st.session_state.text_parsed = False
            if "preprocess_images" not in st.session_state:
                st.session_state.preprocess_images = []

            # Display original image with info
            st.image(
                original_image,
                caption="Uploaded Image",
                width=300,
            )

            # Preprocess Section
            st.markdown("---")
            st.markdown("### STEP 1 🔍 Image Preprocessing")
            st.markdown("""
            Preprocessing helps improve text extraction accuracy by:
            - Converting to grayscale to reduce complexity
            - Applying noise reduction
            - Enhancing contrast
            - Sharpening details
            - Thresholding for binary image
            """)

            if st.button("🔄 Start Preprocessing"):
                with st.spinner("Applying image preprocessing techniques..."):
                    time.sleep(2)
                    extracted_text, preprocess_images = extract_text_from_image(
                        image_cv
                    )
                    st.session_state.preprocess_images = preprocess_images
                    st.session_state.extracted_text = extracted_text

            # Display preprocessed images with explanations
            step_names = [
                (
                    "Grayscale",
                    "Converts image to black and white for simpler processing",
                ),
                ("Blurred", "Reduces noise and smooths the image"),
                ("Enhanced", "Improves contrast and clarity"),
                ("Sharpened", "Emphasizes text edges"),
                ("Thresholded", "Creates binary (black and white) image"),
            ]

            if st.session_state.preprocess_images:
                st.markdown("### STEP 2 🖼️ Preprocessing Steps")

                for i in range(0, len(step_names), 3):
                    cols = st.columns(3)
                    for col, (step_name, description), img in zip(
                        cols,
                        step_names[i : i + 3],
                        st.session_state.preprocess_images[i : i + 3],
                    ):
                        with col:
                            if img is not None:
                                st.image(
                                    img,
                                    caption=f"{step_name}",
                                    use_column_width=True,
                                )
                                st.info(description)

            # Text Extraction Section
            if st.session_state.preprocess_images:
                st.markdown("---")
                st.markdown("### STEP 3 📝 Text Extraction")
                st.markdown("""
                Using Tesseract OCR engine to detect and extract text from the processed image.
                The engine:
                - Identifies text regions
                - Recognizes individual characters
                - Combines them into words and lines
                """)

                if st.button("📋 Extract Text"):
                    with st.spinner("Running OCR process..."):
                        time.sleep(2)
                        extracted_text = st.session_state.extracted_text
                        logging.info(f"Extracted data: {extracted_text}")

                    if extracted_text:
                        st.session_state.text_extracted = True
                        st.session_state.text_parsed = False

                if st.session_state.text_extracted and st.session_state.extracted_text:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(
                            original_image,
                            caption="Original Image",
                            use_column_width=True,
                        )
                    with col2:
                        st.subheader("📄 Extracted Text")
                        st.markdown(f"```\n{st.session_state.extracted_text}\n```")

            # Data Parsing Section
            if st.session_state.text_extracted:
                st.markdown("---")
                st.markdown("### STEP 4 🔍 Data Parsing")
                st.markdown("""
                Using regular expressions (regex) to identify and extract specific patterns:
                - Transaction No
                - Transaction Date
                - Transaction Type
                - Sender Name
                - Amount
                - Receiver Name
                - Notes
                """)

                if st.button("🔍 Parse Data"):
                    with st.spinner("Analyzing text patterns..."):
                        time.sleep(2)
                        transaction_details = extract_transaction_data(
                            st.session_state.extracted_text
                        )

                    col1, col2 = st.columns([4, 3])
                    with col1:
                        st.markdown("#### 📑 Raw Extracted Text")
                        st.markdown(f"```\n{st.session_state.extracted_text}\n```")

                    with col2:
                        st.markdown("#### 📊 Structured Data")
                        if transaction_details:
                            for key, value in transaction_details.items():
                                st.markdown(f"**{key}:** {value}")
                            st.session_state.text_parsed = True
                        else:
                            st.warning(
                                "⚠️ No structured data could be parsed from the text."
                            )
                            st.session_state.text_parsed = False

        except Exception as e:
            st.error(f"⚠️ Error processing the image: {e}")

    # Add footer with additional information
    st.markdown("---")
    st.markdown("""
    ### 📚 Additional Information
    - This demo uses OpenCV for image preprocessing
    - Text extraction is powered by Tesseract OCR
    - Regular expressions are used for structured data extraction
    
    For best results, ensure your images are clear and well-lit. The system works best with printed text rather than handwritten content.
    """)
