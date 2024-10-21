import io
import logging
import os
import re

import cv2
import gdown
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from utils.classification import predict_class
from utils.text_extraction import extract_text_from_image, extract_transaction_data

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

# Regular expression patterns for extracting fields
transtype_pattern = re.compile(r"^(Transaction Type|Type)\s?:?\s?(.+)")
notes_pattern = re.compile(r"^(Notes|Note|Purpose|Reason|Remarks)\s?:?\s?(.*)")
transtime_pattern = re.compile(
    r"^(Transaction Time|Date and Time|Date & Time|Transaction Date)\s?:?\s?(.+)"
)
transno_pattern = re.compile(r"^(Transaction No.|Transaction ID)\s?:?\s?(.+)")
receiver_pattern = re.compile(r"^(To|Receiver Name|Send To|Transfer To)\s?:?\s?(.+)")
sender_pattern = re.compile(r"^(From|Sender Name|Send From|Transfer From)\s?:?\s?(.+)")
amount_data_pattern = re.compile(
    r"^(Amount|Total Amount|Total)\s*[:\-–—]?\s*(.+)"
)  # [:\-–—]?: Matches an optional colon, dash, en dash, or em dash.
amount_only_pattern = re.compile(r"(\d*(?:,\d*)*(?:\.\d*)?)\s?(MMK|Ks)$")

class_labels = ["AYAPay", "CBPay", "KPay", "Other", "WavePay"]


@st.cache_resource
def load_model(model_path: str):
    """
    Load and cache the pre-trained model.

    Args:
        model_path: Path to the model weights file.

    Returns:
        The loaded model.
    """
    # Load the VGG model with pre-trained weights
    model = models.vgg16_bn(pretrained=True)
    # Update the classifier layer to match the number of class labels
    model.classifier[6] = nn.Linear(4096, len(class_labels))
    # Load your trained model weights
    model.load_state_dict(torch.load(model_path))
    # Set the model to evaluation mode
    model.eval()
    return model


# Use the cached model
model = load_model(model_dir)


def main():
    st.title("Transaction Details Extractor")

    # Initialize session state for transaction details
    if "all_transaction_details" not in st.session_state:
        st.session_state.all_transaction_details = []

    uploaded_files = st.file_uploader(
        "Upload image files", accept_multiple_files=True, type=["png", "jpg", "jpeg"]
    )

    # Track processed files to avoid duplicates
    processed_files = {
        detail["image_file"] for detail in st.session_state.all_transaction_details
    }

    for uploaded_file in uploaded_files:
        if uploaded_file.name not in processed_files:
            try:
                # Read the image file
                image = Image.open(uploaded_file)
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # Extract text from the image
                extracted_text = extract_text_from_image(image_cv)
                logging.info(f"Extracted data from {extracted_text}")

                if extracted_text:
                    # Extract transaction details using regex
                    transaction_details = extract_transaction_data(extracted_text)
                    transaction_details["image_file"] = uploaded_file.name

                    # Predict payment type using image classification
                    transaction_details["Payment Type"] = predict_class(
                        model, uploaded_file, class_labels
                    )

                    # Append the extracted details to the session state list
                    st.session_state.all_transaction_details.append(transaction_details)
                else:
                    st.warning(f"Failed to extract text from {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

    # Display all transaction details in a DataFrame with custom column names
    if st.session_state.all_transaction_details:
        df = pd.DataFrame(st.session_state.all_transaction_details)

        # Reorder columns to have 'Amount' at the end
        df = df[
            [
                "Transaction Date",
                "Transaction No",
                "Sender Name",
                "Receiver Name",
                "Notes",
                "image_file",
                "Amount",
                "Payment Type",
            ]
        ]

        # Rename columns for display
        df.columns = [
            "Transaction Date",
            "Transaction Number",
            "Sender",
            "Receiver",
            "Notes",
            "Image File",
            "Amount",
            "Payment Type",
        ]

        # Convert 'Amount' to numeric, handling any non-numeric values
        df["Amount"] = pd.to_numeric(
            df["Amount"].replace(r"[^\d.]", "", regex=True), errors="coerce"
        )

        # Configure the DataFrame display with column_config
        st.dataframe(
            df,
            column_config={
                "Transaction Date": "Date",
                "Transaction Number": "Number",
                "Sender": "From",
                "Receiver": "To",
                "Notes": "Notes",
                "Image File": "File",
                "Amount": st.column_config.NumberColumn(
                    "Amount",
                    help="Transaction amount in currency",
                    format="%.2f",  # Format to two decimal places
                    min_value=0.0,  # Minimum value constraint
                ),
                "Payment Type": "Type",
            },
            hide_index=True,
        )

        # Calculate and display the total amount
        total_amount = df["Amount"].sum()
        st.write(f"**Total Amount:** {total_amount}")

        # Export DataFrame to Excel
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Transactions")

        # Provide a download button for the Excel file
        st.download_button(
            label="Download Excel",
            data=excel_buffer,
            file_name="transaction_details.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.success("All transaction details have been extracted successfully.")


if __name__ == "__main__":
    main()
