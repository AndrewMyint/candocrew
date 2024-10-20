import io
import logging
import os
import re

import cv2
import gdown
import numpy as np
import pandas as pd
import pytesseract as pyt
import streamlit as st
from dateutil import parser
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image

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
    model_dir = "model/cnn_b5.h5"
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
    model_dir = "model/cnn_b5.h5"

# Set Tesseract command path if needed
# pyt.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Uncomment and set path if necessary

# Regular expression patterns for extracting fields
transtype_pattern = re.compile(r"^(Transaction Type|Type)\s?:?\s?(.+)")
notes_pattern = re.compile(r"^(Notes|Note|Purpose|Reason)\s?:?\s?(.+)")
transtime_pattern = re.compile(
    r"^(Transaction Time|Date and Time|Date & Time|Transaction Date)\s?:?\s?(.+)"
)
transno_pattern = re.compile(r"^(Transaction No|Transaction ID)\s?:?\s?(.+)")
receiver_pattern = re.compile(r"^(To|Receiver Name|Send To)\s?:?\s?(.+)")
sender_pattern = re.compile(r"^(From|Sender Name|Send From)\s?:?\s?(.+)")
amount_data_pattern = re.compile(r"^(Amount|Total Amount)\s?:?\s?(.+)")
amount_only_pattern = re.compile(r"(\d*(?:,\d*)*(?:\.\d*)?)\s?(MMK|Ks)$")

# Classification model path
model_dir = "model/cnn_b5.h5"
class_labels = ["AYAPay", "CBPay", "KPay", "Other", "WavePay"]


@st.cache_resource
def load_cached_model(model_path: str):
    """Load and cache the model."""
    return load_model(model_path)


# Load the pre-trained model
loaded_model = load_cached_model(model_dir)


def extract_text_from_image(image):
    """
    Extracts text from an image using Tesseract OCR.

    :param image: Image file
    :return: Extracted text as a string, or None if extraction fails
    """
    try:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian blur to reduce noise and smoothen the image
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Increase contrast using adaptive histogram equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(blurred)

        # Sharpen the image to make text more readable
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced_img, -1, kernel)

        # Apply a threshold to convert the image to binary (black and white)
        _, thresh = cv2.threshold(
            sharpened, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Convert back to a PIL image
        pil_image = Image.fromarray(thresh)

        # Use Tesseract to do OCR on the image
        config = "--psm 6"
        text = pyt.image_to_string(pil_image, config=config, lang="eng")
        return text

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return None


def split_text_into_lines(text):
    """
    Splits the extracted text into lines.

    :param text: Extracted text
    :return: List of non-empty lines
    """
    lines = text.split("\n")
    return [line.strip() for line in lines if line.strip()]


def extract_date_time(date_time_str):
    """
    Extracts date and time from the input string using regex and dateutil parser.

    :param date_time_str: String containing date and time
    :return: Formatted date and time
    """
    date_pattern = re.compile(
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2} \w+ \d{4}|\w+ \d{1,2}, \d{4})"
    )
    time_pattern = re.compile(
        r"\b((1[0-2]|0?[1-9]):[0-5][0-9](?::[0-5][0-9])?\s?[APap][Mm]|(2[0-3]|[01]?[0-9]):[0-5][0-9](?::[0-5][0-9])?)\b"
    )

    try:
        date_match = date_pattern.search(date_time_str)
        times_match = time_pattern.search(date_time_str)

        formatted_date = (
            parser.parse(date_match.group()).strftime("%Y/%m/%d") if date_match else ""
        )
        formatted_time = (
            parser.parse(times_match.group()).strftime("%H:%M:%S")
            if times_match
            else ""
        )

    except Exception as e:
        logging.error(f"Error parsing date or time: {e}")
        formatted_date, formatted_time = "", ""

    return formatted_date, formatted_time


def extract_amount_only(amount_str):
    """
    Extracts numeric amount from the amount string using regex.

    :param amount_str: amount with negative sign, MMK, Ks
    :return: numeric amount as a string
    """
    amount_only_pattern = re.compile(r"-?\d*(?:,\d*)*(?:\.\d{2})?")
    amount_pattern_match = amount_only_pattern.search(amount_str)

    return (
        amount_pattern_match.group().replace("-", "").strip()
        if amount_pattern_match
        else amount_str
    )


def extract_transaction_data(text):
    """
    Extracts transaction details from the given text.

    :param text: Text extracted from an image
    :return: Dictionary of extracted transaction details
    """
    transaction_data = {
        "Transaction No": None,
        "Transaction Date": None,
        "Transaction Type": None,
        "Sender Name": None,
        "Amount": None,
        "Receiver Name": None,
        "Notes": None,
    }
    lines = split_text_into_lines(text)
    for line in lines:
        # Transaction Time
        if re.search(transtime_pattern, line):
            transtime_pattern_match = transtime_pattern.search(line)
            date_time_str = transtime_pattern_match.group(2).strip()
            transaction_data["Transaction Date"], _ = extract_date_time(date_time_str)

        # Transaction No
        elif re.search(transno_pattern, line):
            transno_pattern_match = transno_pattern.search(line)
            transaction_data["Transaction No"] = transno_pattern_match.group(2).strip()

        # Transaction Type
        elif re.search(transtype_pattern, line):
            transtype_pattern_match = transtype_pattern.search(line)
            transaction_data["Transaction Type"] = transtype_pattern_match.group(
                2
            ).strip()

        # Amounts
        elif re.search(amount_data_pattern, line):
            amount_data_pattern_match = amount_data_pattern.search(line)
            amount_string = amount_data_pattern_match.group(2).strip()
            transaction_data["Amount"] = extract_amount_only(amount_string)

        # Sender Name
        elif re.search(sender_pattern, line):
            sender_pattern_match = sender_pattern.search(line)
            transaction_data["Sender Name"] = sender_pattern_match.group(2).strip()

        # Receiver Name
        elif re.search(receiver_pattern, line):
            receiver_pattern_match = receiver_pattern.search(line)
            transaction_data["Receiver Name"] = receiver_pattern_match.group(2).strip()

        # Notes
        elif re.search(notes_pattern, line):
            notes_match = notes_pattern.search(line)
            transaction_data["Notes"] = notes_match.group(2).strip()

        # Amount (if Amount Field does not exist.)
        elif re.search(amount_only_pattern, line):
            amount_only_pattern_match = amount_only_pattern.search(line)
            amount_only_extracted = (
                amount_only_pattern_match.group(1).replace("-", "").strip()
            )
            if transaction_data["Amount"] is None:
                transaction_data["Amount"] = amount_only_extracted

    return transaction_data


def load_and_preprocess_image(image_path, target_size=(270, 270)):
    # Load the image
    image = load_img(image_path, target_size=target_size)
    # Convert the image to a numpy array
    image = img_to_array(image)
    # Expand dimensions to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    # Normalize the image (if your model was trained on normalized data)
    image /= 255.0
    return image


def predict_class(img_path, class_labels):
    # Preprocess the image
    image = load_and_preprocess_image(img_path)
    # Make predictions
    predictions = loaded_model.predict(image)
    # Process the predictions
    predicted_class = np.argmax(predictions, axis=1)
    predicted_class = class_labels[predicted_class[0]]
    return predicted_class


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
                        uploaded_file, class_labels
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
