import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


def processing_image(image_path):
    # Load the image
    img = Image.open(image_path)

    img = img.convert("RGB")

    # Define preprocessing steps
    img_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Apply preprocessing
    img_transform = img_transform(img)  # torch.Size([3, 224, 224])
    inputs = img_transform.unsqueeze(0)  # torch.Size([1, 3, 224, 224])

    return inputs


def predict_class(model, uploaded_file, class_names):
    """
    Predict the class of an image using a pre-trained model.

    Args:
        model: The pre-trained model for classification.
        uploaded_file: A file-like object containing the image.
        class_names: A list of class names corresponding to model outputs.

    Returns:
        The predicted class name.
    """
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Switch to evaluation mode
    model.eval()

    with torch.no_grad():
        # Preprocess the image
        input_data = processing_image(uploaded_file).to(device)

        # Check input shape
        print("Input data shape:", input_data.shape)  # Debugging line

        # Make predictions
        predicted = model(input_data)

        # Get the predicted class
        probabilities = F.softmax(predicted, dim=1)

        # After applying softmax, the probabilities do not sum to 1. Normalize them
        predicted_class_index = torch.max(probabilities, dim=1)[1]

        # Convert index to class name
        label_name = class_names[predicted_class_index.item()]

        # Display the image
        img = Image.open(uploaded_file)
        image_rgb = np.array(img.convert("RGB"))
        plt.imshow(image_rgb)
        plt.axis("off")  # Hide axis
        plt.show()

        return label_name
