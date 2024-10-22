import logging
import time

import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

from utils.constant import CLASS_LABELS

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def render(model):
    st.markdown("""
        ## 🖼️ Image Classification
        Using VGG16 with Batch Normalization
    """)

    # # Supported Payment Methods
    # st.markdown(
    #     """
    # <div style='margin: 1em 0;'>
    #     <span style='font-weight: bold;'>Supported Payment Methods:</span>
    #     <span class='payment-badge'> KBZPay</span>
    #     <span class='payment-badge'> AYAPay</span>
    #     <span class='payment-badge'> CBPay</span>
    #     <span class='payment-badge'> WaveMoney</span>
    # </div>
    # """,
    #     unsafe_allow_html=True,
    # )

    # Model Architecture Information
    with st.expander("ℹ️ About the Model"):
        st.markdown("""
            ### Model Architecture
            We use a VGG16 architecture with batch normalization, which is:
            - A deep convolutional neural network
            - 16 layers deep (13 convolutional layers + 3 fully connected layers)
            - Enhanced with batch normalization for better training stability
            
            ### Key Features
            - **Batch Normalization**: Normalizes the input of each layer, reducing internal covariate shift
            - **Transfer Learning**: Pre-trained on ImageNet and fine-tuned for our specific classes
            - **Input Size**: 224x224 pixels with 3 color channels (RGB)
        """)

    st.markdown("---")

    # File Upload Section with Enhanced UI
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Select an image file (PNG, JPG, or JPEG)",
        accept_multiple_files=False,
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file:
        try:
            # Read and display the image
            original_image = Image.open(uploaded_file)
            st.image(
                original_image,
                caption="Uploaded Image",
                width=300,
            )

            # Initialize session state
            if "predicted_class" not in st.session_state:
                st.session_state.predicted_class = None

            # Detailed Process Explanation
            st.markdown("### 🔄 Classification Process")
            with st.expander("View Detailed Process"):
                st.markdown("""
                    #### 1. Image Preprocessing
                    - **Resize**: Image is resized to 224x224 pixels
                    - **Normalization**: Pixel values are normalized using ImageNet statistics
                        - Mean: [0.485, 0.456, 0.406]
                        - Std: [0.229, 0.224, 0.225]
                    
                    #### 2. Model Architecture
                    The VGG16-BN processes the image through:
                    - Multiple convolutional layers with 3x3 filters
                    - Batch normalization after each conv layer
                    - Max pooling layers
                    - Three fully connected layers
                    
                    #### 3. Output Interpretation
                    - Model outputs confidence scores for each class
                    - Softmax function converts scores to probabilities
                    - Highest probability determines the predicted class
                """)

            # Classification Section
            st.markdown("### 🎯 Classification")
            if st.button("Classify Image", key="classify_btn"):
                with st.spinner("Processing image through VGG16-BN..."):
                    time.sleep(2)
                    device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    model = model.to(device)
                    model.eval()

                    with torch.no_grad():
                        # Preprocessing pipeline
                        img_transform = transforms.Compose(
                            [
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                ),
                            ]
                        )

                        # Transform and predict
                        img_transform = img_transform(original_image)
                        input_data = img_transform.unsqueeze(0)
                        predicted = model(input_data)

                        # Process and display results
                        scores = predicted.cpu().numpy().flatten()
                        df = pd.DataFrame(
                            scores, index=CLASS_LABELS, columns=["Confidence"]
                        )

                        # Display confidence scores
                        st.markdown("### 📊 Confidence Scores")
                        st.bar_chart(df)

                        # Get and display predicted class
                        _, pred_label = torch.max(predicted, 1)
                        label_name = CLASS_LABELS[pred_label.item()]
                        st.session_state.predicted_class = label_name
                        logging.info(f"Predicted class: {label_name}")

            # Display prediction result
            if st.session_state.predicted_class:
                st.markdown("### 🎉 Result")
                st.success(f"Predicted Class: **{st.session_state.predicted_class}**")

            # Technical Details
            with st.expander("🔍 Technical Details"):
                # Create the technical details text with proper formatting
                technical_details = f"""
                    ### Model Parameters
                    - **Architecture**: VGG16 with Batch Normalization
                    - **Input Resolution**: 224x224 pixels
                    - **Number of Classes**: {len(CLASS_LABELS)}
                    - **Device**: {str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))}
                """
                st.markdown(technical_details)

        except Exception as e:
            st.error(f"⚠️ Error processing the image: {e}")
            logging.error(f"Error during image processing: {e}")
