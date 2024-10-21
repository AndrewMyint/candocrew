import logging
import re
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from utils.text_extraction import extract_text_from_image

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_text_with_regex(text):
    """
    Parses text using regular expressions for demonstration purposes.
    """
    # Example regex patterns for demonstration
    patterns = {
        "Transaction No": r"Transaction No\.?:?\s?(\w+)",
        "Date": r"Date\s?:?\s?(\d{4}-\d{2}-\d{2})",
        "Amount": r"Amount\s?:?\s?(\d+\.\d{2})",
    }
    parsed_data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            parsed_data[key] = match.group(1)
    return parsed_data


def render():
    st.title("Image to Text Extraction Demo")

    uploaded_file = st.file_uploader(
        "Upload an image file", accept_multiple_files=False, type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        try:
            # Read the image file
            image = Image.open(uploaded_file)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Display the original image with a fixed size
            st.image(image, caption="Uploaded Image", use_column_width=False, width=300)

            # Initialize session state for buttons
            if "extracted_text" not in st.session_state:
                st.session_state.extracted_text = None
            if "text_extracted" not in st.session_state:
                st.session_state.text_extracted = False
            if "text_parsed" not in st.session_state:
                st.session_state.text_parsed = False

            # Step 1: Extract Text
            if st.button("Extract Text"):
                with st.spinner("Extracting text..."):
                    time.sleep(2)  # Simulate a delay for demonstration
                    extracted_text = extract_text_from_image(image_cv)
                    logging.info(f"Extracted data: {extracted_text}")

                if extracted_text:
                    st.session_state.extracted_text = extracted_text
                    st.session_state.text_extracted = True
                    st.session_state.text_parsed = False  # Reset parsing status

            # Always display the extracted text if it exists in session state
            if st.session_state.extracted_text:
                st.subheader("Extracted Text")
                st.markdown(f"```\n{st.session_state.extracted_text}\n```")

            # Step 2: Parse Text with Regex
            if st.session_state.text_extracted:
                if st.button("Parse Text with Regex"):
                    with st.spinner("Parsing text..."):
                        time.sleep(1)  # Simulate a delay for parsing
                        transaction_details = parse_text_with_regex(
                            st.session_state.extracted_text
                        )

                    st.subheader("Parsed Transaction Details")
                    if transaction_details:
                        for key, value in transaction_details.items():
                            st.markdown(f"**{key}:** {value}")
                        st.session_state.text_parsed = True
                    else:
                        st.warning("No details parsed from the text.")
                        st.session_state.text_parsed = False
        except Exception as e:
            st.error(f"Error processing the image: {e}")
