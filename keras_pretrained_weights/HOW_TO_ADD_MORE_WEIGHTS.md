# How to Download Additional Model Weights

## ✅ **Currently Downloaded (638 MB)**

You already have these models ready for Colab:
1. **VGG16** (56 MB)
2. **VGG19** (76 MB)
3. **ResNet50** (90 MB)
4. **ResNet101** (164 MB)
5. **ResNet152** (224 MB)
6. **DenseNet121** (28 MB)

**Location:** `keras_pretrained_weights/` folder

---

## 📥 **To Download More Weights (If Needed)**

### **Option 1: Download in Google Colab (Recommended)**

Once you have Drive mounted, any new model will be automatically cached to Drive:

```python
# In Colab - after mounting Drive and setting KERAS_HOME
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

# This downloads once and saves to your Drive
model = MobileNetV2(weights='imagenet', include_top=False)  
model = EfficientNetB0(weights='imagenet', include_top=False)

# Future loads are instant from Drive!
```

### **Option 2: Download Locally Then Upload**

Run this simple Python script locally:

```python
# download_more_weights.py
from tensorflow.keras.applications import (
    DenseNet169, DenseNet201,
    MobileNet, MobileNetV2,
    EfficientNetB0, EfficientNetB1
)

models_to_download = {
    'DenseNet169': DenseNet169,
    'DenseNet201': DenseNet201,
    'MobileNet': MobileNet,
    'MobileNetV2': MobileNetV2,
    'EfficientNetB0': EfficientNetB0,
    'EfficientNetB1': EfficientNetB1,
}

for name, Model in models_to_download.items():
    print(f"Downloading {name}...")
    Model(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    print(f"✓ {name} done")

print("All weights downloaded to ~/.keras/models/")
```

Then rerun: `./prepare_weights_for_drive.sh` to repackage

---

## 📋 **Available Models You Can Download**

| Model Family | Models Available |
|--------------|------------------|
| **VGG** | ✅ VGG16, ✅ VGG19 |
| **ResNet** | ✅ ResNet50, ✅ ResNet101, ✅ ResNet152, ResNet50V2, ResNet101V2, ResNet152V2 |
| **DenseNet** | ✅ DenseNet121, DenseNet169, DenseNet201 |
| **MobileNet** | MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large |
| **EfficientNet** | B0, B1, B2, B3, B4, B5, B6, B7 |
| **Inception** | InceptionV3, InceptionResNetV2, Xception |
| **NASNet** | NASNetLarge, NASNetMobile |

✅ = Already downloaded

---

## 💾 **Current Setup is Perfect For:**

The 6 models you have (638 MB) are:
- ✅ Most commonly used architectures
- ✅ Cover different model sizes (VGG, ResNet, DenseNet)
- ✅ Good for experimentation
- ✅ Upload once to Drive, use forever

**You can always add more models later** using Option 1 (download in Colab) - it's easy!

---

## 🚀 **Quick Start**

1. **Upload to Drive:**
   - Drag `keras_pretrained_weights/` folder to Google Drive

2. **Use in Colab:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   import os
   os.environ['KERAS_HOME'] = '/content/drive/MyDrive/keras_pretrained_weights'
   
   # Now use models - they load instantly from Drive!
   from tensorflow.keras.applications import ResNet50
   model = ResNet50(weights='imagenet', include_top=False)
   ```

3. **Done!** No re-downloading needed 🎉
