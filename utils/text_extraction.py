import logging
import re

import cv2
import numpy as np
import pytesseract as pyt
from dateutil import parser
from PIL import Image
from textblob import TextBlob

# Regular expression patterns for extracting fields
patterns = {
    "transaction_type": re.compile(r"^(Transaction Type|Type)\s?:?\s?(.+)"),
    "notes": re.compile(r"^(Notes|Note|Purpose|Reason|Remarks)\s?:?\s?(.*)"),
    "transaction_time": re.compile(
        r"^(Transaction Time|Date and Time|Date & Time|Transaction Date)\s?:?\s?(.+)"
    ),
    "transaction_no": re.compile(
        r"^(Transaction No.|Transaction ID|Transaction Code)\s?:?\s?(.+)"
    ),
    "receiver": re.compile(r"^(Receiver Name|Send To|Transfer To)\s?:?\s?(.+)"),
    "sender": re.compile(r"^(Sender Name|Send From|Transfer From)\s?:?\s?(.+)"),
    "amount_data": re.compile(
        r"^(Amount|Total Amount|Total)\s*[:\-–—]?\s*(.+)"
    ),  # [:\-–—]?: Matches an optional colon, dash, en dash, or em dash.
    "amount_only": re.compile(r"(\d*(?:,\d*)*(?:\.\d*)?)\s?(MMK|Ks)$"),
}


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
        clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8, 8))

        logging.info(f"Enhancing image using CLAHE {clahe}")

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

        config = "--psm 6 --oem 3"
        # config = "--psm 6"
        # Use Tesseract to do OCR on the image
        text = pyt.image_to_string(pil_image, config=config, lang="eng")
        return text, [
            gray,
            blurred,
            enhanced_img,
            sharpened,
            thresh,
        ]

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


MONTH_CORRECTIONS = {
    "Janury": "January",
    "Februry": "February",
    "Marh": "March",
    "Aplil": "April",
    "Mayy": "May",
    "Juen": "June",
    "Jully": "July",
    "Agust": "August",
    "Septmber": "September",
    "Octaber": "October",
    "Novmber": "November",
    "Decmber": "December",
}


def correct_month_in_string(date_string):
    for incorrect_month, correct_month in MONTH_CORRECTIONS.items():
        if incorrect_month in date_string:
            return date_string.replace(incorrect_month, correct_month)
    return date_string


def spell_check_string(text):
    corrected_text = str(TextBlob(text).correct())
    return corrected_text


def extract_date_time(date_time_str):
    """
    Extracts date and time from the input string using regex and dateutil parser.

    :param date_time_str: String containing date and time
    :return: Formatted date and time
    """

    date_time_str = correct_month_in_string(date_time_str)
    date_time_str = spell_check_string(date_time_str)

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

        # Loop through each pattern in the patterns dictionary
        for field, pattern in patterns.items():
            match = pattern.search(normalized_line)

            if match:
                if field == "transaction_time":
                    date_time_str = match.group(2).strip()
                    transaction_data["Transaction Date"], _ = extract_date_time(
                        date_time_str
                    )
                    logging.info(
                        f"Extracted Transaction Date: {transaction_data['Transaction Date']}"
                    )

                elif field == "transaction_no":
                    transaction_data["Transaction No"] = match.group(2).strip()
                    logging.info(
                        f"Extracted Transaction No: {transaction_data['Transaction No']}"
                    )

                elif field == "transaction_type":
                    transaction_data["Transaction Type"] = match.group(2).strip()
                    logging.info(
                        f"Extracted Transaction Type: {transaction_data['Transaction Type']}"
                    )

                elif field == "amount_data":
                    amount_string = match.group(2).strip()
                    transaction_data["Amount"] = extract_amount_only(amount_string)
                    logging.info(f"Extracted Amount: {transaction_data['Amount']}")

                elif field == "sender":
                    transaction_data["Sender Name"] = match.group(2).strip()
                    logging.info(
                        f"Extracted Sender Name: {transaction_data['Sender Name']}"
                    )

                elif field == "receiver":
                    transaction_data["Receiver Name"] = match.group(2).strip()
                    logging.info(
                        f"Extracted Receiver Name: {transaction_data['Receiver Name']}"
                    )

                elif field == "notes":
                    notes_content = match.group(2).strip()
                    transaction_data["Notes"] = notes_content or None
                    logging.info(f"Extracted Notes: {transaction_data['Notes']}")

                elif field == "amount_only" and transaction_data["Amount"] is None:
                    transaction_data["Amount"] = match.group(1).replace("-", "").strip()
                    logging.info(
                        f"Extracted Amount (from amount only pattern): {transaction_data['Amount']}"
                    )

    return transaction_data
