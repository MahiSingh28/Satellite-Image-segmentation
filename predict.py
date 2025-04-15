import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from segmentation_models_pytorch import Unet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#LOAD THE MODEL 
model = Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)
checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

#PREPROCESS THE IMAGE
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    original_width, original_height = image.size

    # Calculate padding to make dimensions divisible by 32
    pad_width = (32 - original_width % 32) % 32
    pad_height = (32 - original_height % 32) % 32

    # Apply padding (right, bottom only)
    padding = (0, 0, pad_width, pad_height)  # left, top, right, bottom
    padded_image = transforms.functional.pad(image, padding, fill=0)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(padded_image).unsqueeze(0)  # Add batch dimension


#Predict the segmentation mask
def predict(image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        return (mask > 0.5).astype(np.uint8)  # Binarize


#Visualize and save
def save_and_show(mask, save_path="predicted_mask.png"):
    plt.imsave(save_path, mask, cmap='gray')
    plt.imshow(mask, cmap='gray')
    plt.title("Predicted Road Segmentation")
    plt.axis("off")
    plt.show()


#Run everything
if __name__ == "__main__":
    image_tensor = preprocess_image("test_10.png")
    mask = predict(image_tensor)
    save_and_show(mask)
