# 🧩 Face Analysis System — Mask Detection & Age Group Classification

A deep learning–powered **Face Analysis System** that detects **mask usage** 😷 and classifies **age groups** 👶🧒🧑👨‍🦳 in real-time using **CNNs** and **transfer learning** (MobileNetV2 + ResNet50).  
Deployed through a **Streamlit web app**, it allows users to analyze uploaded images or live webcam feeds.

---

## 🌟 Features
- 🔍 Real-time **Mask Detection**
- ⏳ **Age Group Classification** into 5 categories (Child, Teen, Adult, Middle, Senior)
- 🧠 Uses **Transfer Learning** with MobileNetV2 & ResNet50
- 🎥 Works with **Image Uploads** or **Webcam Stream**
- 💻 Streamlit-based Interactive Interface

---

## 🧰 Project Structure
```
Face_Analysis_System/
├── app/
│ └── web_app.py
├── models/ # Downloaded automatically
├── prerequisites/
│ └── download_models.py # Downloads pre-trained models
├── src/
│ ├── preprocess.py
│ ├── train_mask_model.py
│ ├── train_age_model.py
│ └── detect_and_classify.py
├──datasets/ # This directory can be manually downloaded if you wish to train your model yourself.
|  ├── faceToAge/ # This contains 23708 pictures of varying age groups of people
|  └── maskAndNoMask/ # This is divided into train and test datasets
|     ├── test/
|     |    ├──mask
|     |    └──no_mask
|     └── train/
|          ├──mask
|          └──no_mask
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Mr-Johnes/FaceSage.git
cd FaceSage
```

### 2️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

### 3️⃣ Download Pre-Trained Models
```bash
python prerequisites/download_models.py
```

After this step the ```models/``` folder will contain:
```
mask_detector.h5
age_classifier_v2.h5
```

### 4️⃣ Run the Web App
```bash
streamlit run app/web_app.py
```
---

### ⚠️⚠️ Problem with the downloads:

If you face any problem accessing the pre-trained model then use the below link for the Google Drive link:
- ☁️ Drive Link: https://drive.google.com/drive/folders/1CaNXvXBZbH51uXgF5J9i11Q4Ug2zegDM?usp=sharing

If you wish to train your models locally yourself then download the datasets ```faceToAge``` and ```maskAndNoMask``` from the drive link to the ```datasets``` folder.
Then run the following files from the ```src``` folder:
```text
train_mask_model.py
train_age_model.py
```


---

## 📊 Technologies Used

- 🧠 TensorFlow / Keras — Model training & transfer learning
- 🖼 OpenCV — Face detection & preprocessing
- 🌐 Streamlit — Web interface
- ☁️ Google Drive + gdown — Model hosting

---

### Acknowledgement

- Mask Dataset: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset?resource=download
- UTKFace: https://www.kaggle.com/datasets/jangedoo/utkface-new/data

---

## Things to note

### This project is made for research purposes only. The final product is non-commercial and we hold no licence over it.

---
## ✨ Authors

### Abhinav Sah and Raj Aman

- 📧 Developed as part of academic coursework (Face Analysis System Project)
- 💡 “Where AI meets human expression.”
