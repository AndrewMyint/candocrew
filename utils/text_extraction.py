import logging
import re

import cv2
import numpy as np
import pytesseract as pyt
from dateutil import parser
from PIL import Image

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
        logging.info(f"Processing line: {line}")

        # Normalize line
        normalized_line = re.sub(r"\s+", " ", line).strip()
        logging.debug(f"Normalized line: {normalized_line}")

        # Transaction Time
        if re.search(transtime_pattern, normalized_line):
            transtime_pattern_match = transtime_pattern.search(normalized_line)
            date_time_str = transtime_pattern_match.group(2).strip()
            transaction_data["Transaction Date"], _ = extract_date_time(date_time_str)
            logging.info(
                f"Extracted Transaction Date: {transaction_data['Transaction Date']}"
            )

        # Transaction No
        elif re.search(transno_pattern, normalized_line):
            transno_pattern_match = transno_pattern.search(normalized_line)
            transaction_data["Transaction No"] = transno_pattern_match.group(2).strip()
            logging.info(
                f"Extracted Transaction No: {transaction_data['Transaction No']}"
            )

        # Transaction Type
        elif re.search(transtype_pattern, normalized_line):
            transtype_pattern_match = transtype_pattern.search(normalized_line)
            transaction_data["Transaction Type"] = transtype_pattern_match.group(
                2
            ).strip()
            logging.info(
                f"Extracted Transaction Type: {transaction_data['Transaction Type']}"
            )

        # Amounts
        elif re.search(amount_data_pattern, normalized_line):
            amount_data_pattern_match = amount_data_pattern.search(normalized_line)
            amount_string = amount_data_pattern_match.group(2).strip()
            transaction_data["Amount"] = extract_amount_only(amount_string)
            logging.info(
                f"Extracted Amount: {transaction_data['Amount']}, and length: {len(transaction_data['Amount'])}"
            )
            logging.info(f"Amount String: {transaction_data["Amount"] is None}")
            logging.info(f"Amount Type: {type(transaction_data["Amount"])}")

        # Sender Name
        elif re.search(sender_pattern, normalized_line):
            sender_pattern_match = sender_pattern.search(normalized_line)
            transaction_data["Sender Name"] = sender_pattern_match.group(2).strip()
            logging.info(f"Extracted Sender Name: {transaction_data['Sender Name']}")

        # Receiver Name
        elif re.search(receiver_pattern, normalized_line):
            receiver_pattern_match = receiver_pattern.search(normalized_line)
            transaction_data["Receiver Name"] = receiver_pattern_match.group(2).strip()
            logging.info(
                f"Extracted Receiver Name: {transaction_data['Receiver Name']}"
            )

        # Notes
        elif re.search(notes_pattern, line):
            notes_match = notes_pattern.search(line)
            notes_content = notes_match.group(2).strip()
            if notes_content:
                transaction_data["Notes"] = notes_content
            else:
                transaction_data["Notes"] = None
            logging.info(f"Extracted Notes: {transaction_data['Notes']}")

        # Amount (if Amount Field does not exist.)
        elif re.search(amount_only_pattern, normalized_line):
            amount_only_pattern_match = amount_only_pattern.search(normalized_line)
            amount_only_extracted = (
                amount_only_pattern_match.group(1).replace("-", "").strip()
            )
            if transaction_data["Amount"] is None:
                transaction_data["Amount"] = amount_only_extracted
                logging.info(
                    f"Extracted Amount (from amount only pattern): {transaction_data['Amount']}"
                )

    return transaction_data
