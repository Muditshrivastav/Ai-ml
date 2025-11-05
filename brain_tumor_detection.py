# 🧠 Brain Tumor Classification using MobileNetV2

A deep learning project that classifies MRI brain images to detect whether a patient has a **brain tumor** or not.  
This model leverages **transfer learning** using the **MobileNetV2** architecture pretrained on **ImageNet** to achieve high accuracy even with a limited dataset.

---

## 📁 Dataset

**Dataset:** [Brain Tumor Classification (Kaggle)](https://www.kaggle.com/datasets)  
- Contains MRI images of human brains.  
- Two classes:
  - `Tumor` — MRI scans showing the presence of a brain tumor.
  - `No Tumor` — MRI scans showing healthy brain structure.

The dataset was cleaned, augmented, and split into:
- **Train set**
- **Validation set**
- **Test set**

---

## 🧠 Model Architecture

**Model Used:** `MobileNetV2` (pretrained on ImageNet)  
**Framework:** TensorFlow / Keras  

### 🔍 Architecture Details
- Base model: MobileNetV2 (frozen pretrained layers)
- Added custom dense layers for classification:
  - GlobalAveragePooling2D
  - Dense(128, activation='relu')
  - Dropout(0.3)
  - Dense(1, activation='sigmoid')

### ⚙️ Loss & Optimizer
- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Metric:** Accuracy  

---

## 🚀 Training

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
