# 🧠 Brain Tumor Classification using MobileNetV2

This project focuses on detecting **brain tumors** from MRI scan images using **deep learning**.  
The model is built with **MobileNetV2**, a lightweight and efficient convolutional neural network pretrained on **ImageNet**, and fine-tuned on a **brain tumor dataset from Kaggle**.  
It achieves over **90% accuracy** in predicting whether a patient has a tumor or not.

---

## 📂 Dataset

- **Source:** Brain Tumor Classification Dataset from Kaggle  
- **Classes:**  
  - **Tumor** — MRI scans showing tumor presence  
  - **No Tumor** — MRI scans showing healthy brains  
- The dataset contains around **3,000+ images**, divided into training, validation, and testing sets.  
- Images were preprocessed by:
  - Resizing to **224×224 pixels**
  - Normalizing pixel values between 0 and 1
  - Augmenting with flips, rotations, and zooms to improve generalization

---

## 🧠 Model Overview

- **Base Model:** MobileNetV2 (pretrained on ImageNet)  
- **Framework:** TensorFlow / Keras  
- The pretrained base model was used as a **feature extractor**, and additional dense layers were added for binary classification.

### Model Highlights
- The pretrained layers of MobileNetV2 were **frozen** to retain learned visual features.
- Custom layers were added on top for **tumor vs non-tumor classification**.
- A **sigmoid activation** was used in the final layer for binary output.

---

## ⚙️ Training Process

1. **Transfer Learning:**  
   MobileNetV2 was initialized with pretrained ImageNet weights.  
   Only the top (custom) layers were trained initially to avoid overfitting.

2. **Fine-Tuning:**  
   After initial training, a few deeper layers of MobileNetV2 were unfrozen to fine-tune the model on MRI data, improving feature alignment.

3. **Training Details:**  
   - **Loss Function:** Binary Crossentropy  
   - **Optimizer:** Adam (learning rate = 0.0001)  
   - **Batch Size:** 32  
   - **Epochs:** 15–25 (early stopping applied)  
   - **Validation Split:** 20% of training data used for validation  

4. **Evaluation:**  
   The model was evaluated on unseen test data to measure accuracy, precision, recall, and F1-score.

---

## 📊 Results

| Metric | Score |
|:--------|:------|
| **Training Accuracy** | 94% |
| **Validation Accuracy** | 91% |
| **Test Accuracy** | **90%+** |

The model demonstrates strong performance and generalization across unseen MRI images.

---

## 🧩 Key Features

- Uses **transfer learning** for faster convergence and high accuracy  
- Lightweight **MobileNetV2** backbone suitable for deployment  
- Works efficiently even on limited medical datasets  
- Capable of real-time inference in clinical environments

---

## 🔮 Future Work

- Integrate **Grad-CAM** visualization to highlight tumor regions  
- Explore **EfficientNet** and **Vision Transformers (ViT)** for comparison  
- Deploy the model in a **Streamlit-based web app** for patient and doctor use  
- Expand dataset for better robustness and reduce bias

---

## 🧑‍💻 Author

**Mudit Shrivastav**  
🎓 AI & ML Enthusiast | Deep Learning Developer  

Feel free to connect on [LinkedIn](https://www.linkedin.com/in/mudit-shrivastav-81199326a?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3BwaCiXUJBTQO32KUenzKdxw%3D%3D) or contribute to improve the model.

---

## 📜 License

This project is released under the **MIT License**.

