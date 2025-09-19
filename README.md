# Skin Lesion Detection AI 🔬

This is a web application built with Streamlit that uses a Convolutional Neural Network (CNN) to classify common types of skin lesions from user-uploaded images.

---
## Features

* **Image Upload:** Supports JPG, JPEG, and PNG image formats.
* **AI-Powered Classification:** Utilizes a trained TensorFlow/Keras model to predict one of seven types of skin lesions.
* **Confidence Score:** Displays the model's confidence in its prediction.
* **Detailed Information:** Provides a brief, informative description for each predicted lesion type.
* **Clean UI:** A simple, focused, and permanent dark-themed user interface.

---
## Tech Stack

* **Language:** Python
* **Framework:** Streamlit
* **Machine Learning:** TensorFlow / Keras
* **Data Handling:** Pandas, NumPy
* **Image Processing:** Pillow

---
## Setup and Installation

Follow these instructions to get the project running on your local machine.

### Part 1: Running the Application with the Pre-trained Model

These steps will run the web app using the `skin_lesion_classifier.h5` model already included in this repository.

#### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

#### 2. Create a Python Virtual Environment
 ```bash
python3 -m venv venv
source venv/bin/activate
```
#### 3. Install Dependencies
 ```bash
pip install -r requirements.txt
```
#### 4. Run the Streamlit App
 ```bash
streamlit run app.py
```
### Part 2: (Optional) Re-training the Model
If you wish to re-train the model from scratch, follow these additional steps.

#### 1. Download the Dataset
The model was trained on the Skin Cancer MNIST: HAM10000 dataset. You must download it from Kaggle:

Dataset Link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

#### 2. Set Up the Folder Structure
After downloading, place the dataset folders and the metadata CSV file in the root of your project directory. The structure must be as follows for the training script to work:
```bash
your-repo-name/
├── app.py
├── train_model.py
├── skin_lesion_classifier.h5
├── HAM10000_metadata.csv            <-- From Kaggle
├── HAM10000_images_part_1/          <-- From Kaggle
│   └── ... (image files)
└── HAM10000_images_part_2/          <-- From Kaggle
    └── ... (image files)
```
#### 3. Run the Training Script
Execute the training script from your terminal. This process may take some time depending on your hardware.
```bash
python train_model.py
```
This will generate a new skin_lesion_classifier.h5 file in your project directory, which the app.py will automatically use.

## Usage
Once the application is running, navigate to the local URL provided by Streamlit (usually http://localhost:8501).

Click the file uploader to select an image of a skin lesion from your computer.

Click the Predict button.

The results, including the predicted class, confidence score, and information, will appear on the right-hand side.

    
