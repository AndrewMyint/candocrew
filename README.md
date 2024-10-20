# candoall

# Transaction Details Extractor

This project is a Streamlit application designed to extract transaction details from uploaded images using Optical Character Recognition (OCR) with Tesseract. The application also classifies the payment type using a pre-trained model. The extracted data is displayed in a structured format and can be exported to an Excel file.

## Features

- **Image Upload**: Upload multiple image files in PNG, JPG, or JPEG format.
- **OCR Processing**: Extracts text from images using Tesseract OCR.
- **Data Extraction**: Uses regular expressions to extract transaction details such as date, transaction number, amount, sender, receiver, and notes.
- **Image Classification**: Classifies the payment type using a pre-trained model.
- **Data Display**: Displays extracted data in a Streamlit DataFrame with customizable column configurations.
- **Excel Export**: Allows users to download the extracted data as an Excel file.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AndrewMyint/candoall.git
   cd candoall
   ```

2. **Install the required packages**:
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract**:
   - **Windows**: Download the installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
   - **macOS**: Use Homebrew:
     ```bash
     brew install tesseract
     ```
   - **Linux**: Use apt-get:
     ```bash
     sudo apt-get install tesseract-ocr
     ```

4. **Download the Pre-trained Model**:
   - Ensure the pre-trained model file `cnn_b5.h5` is placed in the `model` directory. You may need to download it from a cloud storage service or use Git LFS if it's stored in the repository.

## Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run main.py
   ```

2. **Upload Images**:
   - Use the file uploader in the app to upload images containing transaction details.

3. **View Extracted Data**:
   - The app will display the extracted transaction details in a table format.

4. **Download Excel**:
   - Click the "Download Excel" button to download the extracted data as an Excel file.

## Project Structure

- `main.py`: The main application script.
- `requirements.txt`: List of Python dependencies.
- `model/cnn_b5.h5`: Pre-trained model for payment type classification.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for providing an easy-to-use framework for building web apps.
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for the OCR engine.
- [Pandas](https://pandas.pydata.org/) for data manipulation and analysis.
- [Keras](https://keras.io/) for the deep learning model used in image classification.