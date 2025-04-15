# Satellite-Image-segmentation
Pix2Pix GAN and UNet-based solution for road segmentation from satellite images. Includes data augmentation, training pipeline


---

# 🚀 Satellite Image Road Segmentation

This project applies **Pix2Pix GAN** for data augmentation and **UNet** for road segmentation on satellite images. It includes model training, evaluation, and visualization of segmentation performance.

---

## 📁 Project Structure

```
Satellite-Image-for-road-segmentation/
├── Pix2Pix Generator & Discriminator (GAN)
├── UNet segmentation model (pre trained ResNet34 encoder)
├── Data augmentation using GAN
├── Training scripts
├── Evaluation scripts (F1 score, visualization)
├── Generated image-saving utilities
└── README.md
```

---

## 🧠 Models Used

### 1. **Pix2Pix GAN**
- Generates synthetic satellite images from binary road masks.
- Helps augment limited training data and improve segmentation performance.

### 2. **UNet (with ResNet34 encoder)**
- Performs binary segmentation to detect roads from satellite images.
- Trained on both original and GAN-augmented data.

---

## 🔄 Data Pipeline

1. Load and preprocess the satellite image dataset.
2. Train Pix2Pix GAN on image-mask pairs.
3. Generate new synthetic images using the GAN.
4. Combine generated and original data into an augmented dataset.
5. Train UNet on the augmented dataset.
6. Evaluate the UNet model using F1 score and visual inspection.

---

## 📊 Evaluation

- **F1 Score**: Used to assess segmentation accuracy.
- **Visualization**: Model predictions are displayed alongside ground truth for comparison.
- **Model Saving**: Best model is saved as `best_model.pth`.





---

## ✍️ Author

Created by **Mahi Singh**  
Feel free to connect or explore more on [GitHub](https://github.com/Mahisingh28)

