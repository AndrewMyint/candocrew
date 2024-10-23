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


TEXT_EXTRACTION_PAGE = "Text Extraction"
TRANSACTION_SUMMARY_PAGE = "Transaction Summary"
IMAGE_CLASSIFICATION_PAGE = "Image Classification"

PAGE_DESCRIPTIONS = {
    TEXT_EXTRACTION_PAGE: "Demonstrates image to text extraction using Tesseract OCR, and text parsing with regex.",
    IMAGE_CLASSIFICATION_PAGE: "Classifies images using a pre-trained VGG16 model with batch normalization. This model is fine-tuned to recognize specific classes relevant to the application.",
    TRANSACTION_SUMMARY_PAGE: (
        "Manually adding transaction screenshots? Let our app handle it! "
        "With just two clicks, you can:\n\n"
        "- Extract and Summarize Details from your mobile receipts instantly.\n"
        "- Download a Ready-to-Use Excel Sheet with all your data.\n\n"
        "Save time, reduce errors, and streamline your accounting in seconds!"
    ),
}


def main():
    # Define the pages
    pages = {
        TRANSACTION_SUMMARY_PAGE: transaction_details_extractor,
        TEXT_EXTRACTION_PAGE: text_extraction_page,
        IMAGE_CLASSIFICATION_PAGE: classification_page,
    }

    st.sidebar.markdown("## SNAPSHEET")
    selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Note:** {PAGE_DESCRIPTIONS[selected_page]}")

    # Render the selected page
    if selected_page == TRANSACTION_SUMMARY_PAGE:
        transaction_details_extractor.render(model)
    elif selected_page == TEXT_EXTRACTION_PAGE:
        text_extraction_page.render()
    elif selected_page == IMAGE_CLASSIFICATION_PAGE:
        classification_page.render(model)


if __name__ == "__main__":
    main()
