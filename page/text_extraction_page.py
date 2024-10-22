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
    st.markdown("## Image to Text Extraction")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload an image file (PNG, JPG, or JPEG)",
        accept_multiple_files=False,
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file:
        try:
            # Read the image file
            original_image = Image.open(uploaded_file)
            image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

            # Initialize session state for buttons
            if "extracted_text" not in st.session_state:
                st.session_state.extracted_text = None
            if "text_extracted" not in st.session_state:
                st.session_state.text_extracted = False
            if "text_parsed" not in st.session_state:
                st.session_state.text_parsed = False
            if "preprocess_images" not in st.session_state:
                st.session_state.preprocess_images = []

            st.image(
                original_image,
                caption="Uploaded Image",
                width=300,
            )

            # Preprocess Image
            st.markdown("---")
            if st.button("Preprocess Image"):
                with st.spinner("Preprocessing image..."):
                    time.sleep(2)
                    extracted_text, preprocess_images = extract_text_from_image(
                        image_cv
                    )
                    st.session_state.preprocess_images = preprocess_images
                    st.session_state.extracted_text = extracted_text

            # Display preprocessed images in three columns at a time
            step_names = [
                "Grayscale",
                "Blurred",
                "Enhanced",
                "Sharpened",
                "Thresholded",
            ]

            # Ensure there are preprocessed images available to display
            if st.session_state.preprocess_images:
                st.markdown("## Preprocessed Images")

                # Iterate over the images and display them in three columns per row
                for i in range(0, len(step_names), 3):
                    cols = st.columns(3)  # Create three columns per row

                    # Loop through the columns and corresponding images
                    for col, step_name, img in zip(
                        cols,
                        step_names[i : i + 3],
                        st.session_state.preprocess_images[i : i + 3],
                    ):
                        with col:
                            if img is not None:  # Ensure the image is not empty
                                st.image(
                                    img,
                                    caption=f"{step_name} Image",
                                    use_column_width=True,
                                )

            # Step 1: Extract Text
            if (
                st.session_state.preprocess_images
                and len(st.session_state.preprocess_images) > 0
            ):
                st.markdown("---")
                if st.button("Extract Text"):
                    with st.spinner("Extracting text..."):
                        time.sleep(2)  # Simulate a delay for demonstration
                        # extracted_text, _ = extract_text_from_image(image_cv)
                        extracted_text = st.session_state.extracted_text
                        logging.info(f"Extracted data: {extracted_text}")

                    if extracted_text:
                        # st.session_state.extracted_text = extracted_text
                        st.session_state.text_extracted = True
                        st.session_state.text_parsed = False  # Reset parsing status

                # Display image and extracted text side by side after text extraction
                if st.session_state.text_extracted and st.session_state.extracted_text:
                    col1, col2 = st.columns(
                        [1, 2]
                    )  # Adjust column ratio (1:2 for wider text column)

                    with col1:
                        st.image(
                            original_image,
                            caption="Uploaded Image",
                            use_column_width=True,
                        )

                    with col2:
                        st.subheader("Extracted Text")
                        st.markdown(f"```\n{st.session_state.extracted_text}\n```")

            # Step 2: Parse Text with Regex
            if st.session_state.text_extracted:
                st.markdown("---")
                if st.button("Parse with Regex"):
                    with st.spinner("Parsing text..."):
                        time.sleep(2)  # Simulate a delay for parsing
                        transaction_details = extract_transaction_data(
                            st.session_state.extracted_text
                        )

                    col1, col2 = st.columns([4, 3])
                    with col1:
                        st.markdown("#### Extracted Text")
                        st.markdown(f"```\n{st.session_state.extracted_text}\n```")

                    with col2:
                        st.markdown("#### Transaction Details")
                        if transaction_details:
                            for key, value in transaction_details.items():
                                st.markdown(f"**{key}:** {value}")
                            st.session_state.text_parsed = True
                        else:
                            st.warning("No details parsed from the text.")
                            st.session_state.text_parsed = False

        except Exception as e:
            st.error(f"Error processing the image: {e}")
