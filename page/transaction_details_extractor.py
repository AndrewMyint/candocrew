import base64
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
    # Custom CSS for title styling
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 3em;
            font-weight: bold;
            color: #1F75FE;
        }
        .subtitle {
            font-size: 1.5em;
            color: #666;
            margin-top: -0.5em;
        }
        .payment-badge {
            display: inline-block;
            padding: 0.2em 0.4em;
            margin: 0.2em;
            border-radius: 10px;
            font-weight: bold;
            background-color: #f0f2f6;
            color: #333;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # App Title with styled layout
    st.markdown('<p class="main-title">SNAPSHEET</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Your Go-To Solution for Organizing Mobile Receipts</p>',
        unsafe_allow_html=True,
    )
    # <div style='display: flex; align-items: center; gap: 10px;'>
    #     <img src='https://as2.ftcdn.net/v2/jpg/05/30/99/23/1000_F_530992386_EKRH5bJKgcj68ZI6FAXLS1rzpByqLi6R.jpg' width='40'/>
    #     <h2>Image Classification</h2>
    # </div>

    def img_to_base64(img_path):
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    # Get base64 encoded image
    kpay_path = "assets/images/kbzpaylogo.png"
    aya_path = "assets/images/ayapaylogo.png"
    kpay_logo = img_to_base64(kpay_path)
    aya_logo = img_to_base64(aya_path)

    st.markdown(
        f"""
        <div style='margin: 1em 0;'>
            <span style='font-weight: bold;'>Supported Payment Methods:</span>
            <span class='payment-badge'><img src='data:image/png;base64,{kpay_logo}' width='20' alt='KBZPay Logo'/> KBZPay</span>
            <span class='payment-badge'><img src='data:image/png;base64,{aya_logo}' width='20' alt='KBZPay Logo'/> AYAPay</span>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Hero Section with Motivation
    st.markdown("""
    ### 📱 Simplify Your Mobile Transaction Accounting!
    
    Manually adding transaction screenshots? Let our app handle it! With just **two clicks**, you can:
    * **Extract and Summarize Details** from your mobile receipts instantly
    * **Download a Ready-to-Use Excel Sheet** with all your data
    
    Save time, reduce errors, and streamline your accounting in seconds!
    """)

    # Main upload section with a card-like design
    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📤 Upload Your Transaction Screenshots")
        st.markdown("Supported formats: PNG, JPG, JPEG")

    # Initialize session state for transaction details
    if "all_transaction_details" not in st.session_state:
        st.session_state.all_transaction_details = []

    uploaded_files = st.file_uploader(
        "Drop your files here or click to browse",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg"],
    )

    # Processing Section
    if uploaded_files:
        st.markdown("### 🔄 Processing Your Files")

        # Add a progress bar for visual feedback
        progress_bar = st.progress(0)

        # Track processed files to avoid duplicates
        processed_files = {
            detail["image_file"] for detail in st.session_state.all_transaction_details
        }

        for idx, uploaded_file in enumerate(uploaded_files):
            if uploaded_file.name not in processed_files:
                try:
                    # Update progress
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)

                    # Read the image file
                    image = Image.open(uploaded_file)
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                    # Extract text from the image
                    extracted_text, _ = extract_text_from_image(image_cv)

                    if extracted_text:
                        transaction_details = extract_transaction_data(extracted_text)
                        transaction_details["image_file"] = uploaded_file.name
                        transaction_details["Payment Type"] = predict_class(
                            model, uploaded_file, CLASS_LABELS
                        )
                        st.session_state.all_transaction_details.append(
                            transaction_details
                        )
                    else:
                        st.warning(
                            f"⚠️ Failed to extract text from {uploaded_file.name}"
                        )
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {e}")

    # Results Section
    if st.session_state.all_transaction_details:
        st.markdown("### 📊 Extracted Transaction Details")

        df = pd.DataFrame(st.session_state.all_transaction_details)
        df = df[
            [
                "Transaction Date",
                "Transaction No",
                "Payment Type",
                "Sender Name",
                "Receiver Name",
                "Notes",
                "Amount",
                "image_file",
            ]
        ]

        # Rename and format columns
        df.columns = [
            "Transaction Date",
            "Transaction No",
            "Payment Type",
            "Sender",
            "Receiver",
            "Notes",
            "Amount",
            "Image File",
        ]

        df["Amount"] = pd.to_numeric(
            df["Amount"].replace(r"[^\d.]", "", regex=True), errors="coerce"
        )

        # Display DataFrame with enhanced styling
        st.dataframe(
            df,
            column_config={
                "Transaction Date": st.column_config.DateColumn("Transaction Date"),
                "Transaction No": "Transaction No",
                "Payment Type": "Payment Type",
                "Sender": "Sender",
                "Receiver": "Receiver",
                "Notes": "Notes",
                "Amount": st.column_config.NumberColumn(
                    "Amount",
                    help="Transaction amount in currency",
                    format="%.2f",
                    min_value=0.0,
                ),
                "Image File": "Image File",
            },
            hide_index=True,
        )

        # Summary Section with payment type breakdown
        col1, col2, col3 = st.columns(3)
        with col1:
            total_amount = df["Amount"].sum()
            formatted_amount = f"MMK {total_amount:,.0f}"  # Removed .2f to show full number without decimals
            st.metric("Total Amount", formatted_amount)
        with col2:
            transaction_count = len(df)
            st.metric("Total Transactions", transaction_count)
        with col3:
            payment_breakdown = df["Payment Type"].value_counts()
            st.metric("Payment Methods Used", len(payment_breakdown))

        # Export Section
        st.markdown("### 📥 Export Your Data")
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Transactions")

        st.download_button(
            label="📥 Download Excel Report",
            data=excel_buffer,
            file_name="transaction_details.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # Information Sections
    st.markdown("---")
    with st.expander("ℹ️ What is SnapSheet?"):
        st.markdown("""
        SnapSheet is your intelligent transaction management assistant that automatically processes 
        your KBZPay and AYAPay transaction screenshots and extracts key information including:
        * Transaction dates and numbers
        * Sender and receiver details
        * Transaction amounts
        * Payment types
        * Additional notes
        """)

    with st.expander("🛠️ How Does It Work?"):
        st.markdown("""
        1. **Upload Images**: Simply drag and drop your KBZPay or AYAPay transaction screenshots
        2. **Automatic Processing**: Our AI-powered system extracts text and classifies payment types
        3. **Review & Export**: Check the extracted data and download as an Excel file
        """)

    with st.expander("💡 Why Use SnapSheet?"):
        st.markdown("""
        * **Save Time**: Eliminate manual data entry
        * **Reduce Errors**: Automated extraction ensures accuracy
        * **Easy Organization**: Get structured data ready for accounting
        * **Instant Results**: Process multiple transactions in seconds
        * **Simple Export**: Download organized Excel reports with one click
        * **Payment Support**: Works with both KBZPay and AYAPay transactions
        """)

    st.markdown("---")
    st.markdown("### 🔒 Privacy Notice")
    st.markdown("""
        Your data privacy is our priority. All uploaded images are processed securely 
        and are not stored on our servers after processing is complete.
    """)
