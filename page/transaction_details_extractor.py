import io
import logging

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from utils.classification import predict_class
from utils.constant import CLASS_LABELS
from utils.text_extraction import extract_text_from_image, extract_transaction_data

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def render(model):
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
                extracted_text, _ = extract_text_from_image(image_cv)
                logging.info(f"Extracted data from {extracted_text}")

                if extracted_text:
                    # Extract transaction details using regex
                    transaction_details = extract_transaction_data(extracted_text)
                    transaction_details["image_file"] = uploaded_file.name

                    # Predict payment type using image classification
                    transaction_details["Payment Type"] = predict_class(
                        model, uploaded_file, CLASS_LABELS
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
