import logging
import os

import gdown
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models

from page import (
    classification_page,
    text_extraction_page,
    transaction_details_extractor,
)
from utils.constant import CLASS_LABELS

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


environment = st.secrets["general"]["ENVIRONMENT"]
os.environ["ENVIRONMENT"] = environment

logging.info(f"Loading dependencies... {os.getenv('ENVIRONMENT')}")


# Function to download the model from Google Drive
def download_model_from_drive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    logging.info(f"Downloading model from {url} to {output_path}")
    try:
        gdown.download(url, output_path, quiet=False)
        logging.info("Download completed.")
    except Exception as e:
        logging.error(f"Failed to download model: {e}")


# Set model path based on environment
if environment == "production":
    logging.info("Running in production environment.")
    # model_dir = "model/cnn_b5.h5"
    model_dir = "model/VGG16BatchNorm03.pth"
    if not os.path.exists(model_dir):
        logging.info("Model file does not exist. Preparing to download.")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)

        # Get the Google Drive file ID from Streamlit secrets
        file_id = st.secrets["general"]["MODEL_FILE_ID"]
        if file_id:
            logging.info("Downloading model from Google Drive...")
            download_model_from_drive(file_id, model_dir)
        else:
            raise ValueError("Google Drive file ID is not set in the secrets.")
    else:
        logging.info("Model file already exists. Skipping download.")
else:
    # Local path for development
    # model_dir = "model/cnn_b5.h5"
    model_dir = "model/VGG16BatchNorm03.pth"

# Set Tesseract command path if needed
# pyt.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Uncomment and set path if necessary


@st.cache_resource
def load_cached_model(model_path: str):
    # Load the VGG model with pre-trained weights
    model = models.vgg16_bn(pretrained=True)
    model.classifier[6] = nn.Linear(4096, len(CLASS_LABELS))

    # Load your trained model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Use the cached model
model = load_cached_model(model_dir)


def main():
    # Define the pages
    pages = {
        "Image to Text Extraction Demo": text_extraction_page,
        "Transaction Details Extractor": transaction_details_extractor,
        "Image Classification": classification_page,  # Add the classification page
        "Demo Page": None,
    }

    st.sidebar.markdown("## SNAPSHEET")
    selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))

    page_info = {
        "Image to Text Extraction Demo": "Demonstrates image to text extraction using Tesseract OCR.",
        "Transaction Details Extractor": "Extracts transaction details from images using OCR and displays them.",
        "Image Classification": "Classifies images using a pre-trained model.",  # Add description
        "Demo Page": "This is a demo page. Add your content here.",
    }
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Note:** {page_info[selected_page]}")

    # Render the selected page
    if selected_page == "Transaction Details Extractor":
        transaction_details_extractor.render(model)
    elif selected_page == "Image to Text Extraction Demo":
        text_extraction_page.render()
    elif selected_page == "Image Classification":
        classification_page.render(model)  # Render the classification page
    elif selected_page == "Demo Page":
        st.title("Demo Page")
        st.write("This is a demo page. Add your content here.")


if __name__ == "__main__":
    main()
