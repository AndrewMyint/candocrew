# Transaction Details Extractor

This Streamlit application extracts transaction details from uploaded image files using Optical Character Recognition (OCR) and image classification. It processes images to extract text, identifies transaction details, and predicts the payment type using a pre-trained model.

## Features

- **OCR Processing**: Extracts text from images using Tesseract OCR.
- **Image Classification**: Classifies images to predict payment types using a pre-trained Keras model.
- **Transaction Details Extraction**: Extracts transaction details such as date, number, sender, receiver, amount, and notes.
- **Data Display**: Displays extracted details in a Streamlit DataFrame with configurable columns.
- **Excel Export**: Allows users to download the extracted transaction details as an Excel file.

## Setup

### Prerequisites

- Python 3.7 or higher
- [Streamlit](https://streamlit.io/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and configured
- [gdown](https://pypi.org/project/gdown/) for downloading files from Google Drive
- [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) for model loading

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up secrets**:
   - Create a `.streamlit/secrets.toml` file in the root directory with the following content:
     ```toml
     [general]
     ENVIRONMENT = "production"
     MODEL_FILE_ID = "your_google_drive_file_id"
     ```

4. **Configure Tesseract**:
   - Ensure Tesseract is installed and the path is set in the script if necessary:
     ```python
     # Uncomment and set path if necessary
     # pyt.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
     ```

## Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run main.py
   ```

2. **Upload Image Files**:
   - Use the file uploader to upload image files in PNG, JPG, or JPEG format.

3. **View Extracted Details**:
   - The app will display extracted transaction details in a table format.

4. **Download Excel**:
   - Use the download button to export the transaction details to an Excel file.

## Configuration

- **Environment Variables**: The application uses `st.secrets` to manage sensitive information like the `ENVIRONMENT` and `MODEL_FILE_ID`. Ensure these are correctly set in the `secrets.toml` file.

## Model File

- The current model file can be accessed and downloaded from Google Drive using the following link:
  [Model File](https://drive.google.com/file/d/1aXlpnWGK5PPCuFvu7zJezPAmsR3wmdYe/view?usp=sharing)

## Logging

- The application uses Python's logging module to log information and errors. Logs are displayed in the console.

## Notes

- Ensure that the model file is available in the specified path or is downloaded from Google Drive if running in production mode.
- The application uses Streamlit's caching mechanism to cache the loaded model, improving performance across user sessions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.