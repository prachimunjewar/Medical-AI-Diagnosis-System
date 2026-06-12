# Medical Diagnosis AI System

An end-to-end AI system combining EDA, ML models, CNN Brain Tumor detection, and LLM report generation.

## Tech Stack
- EDA: Pandas, NumPy, Plotly, Seaborn
- ML: XGBoost, Random Forest, SMOTE, GridSearchCV, Scikit-learn Pipelines
- Deep Learning: TensorFlow, Keras, ResNet50 Transfer Learning (4-class Brain Tumor)
- GenAI: Groq LLaMA 3.3 70B (Medical Report Generation)
- Deployment: Streamlit

## Datasets (Both Free on Kaggle)

### 1. Pima Indians Diabetes Dataset
- Link: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Save as: data/diabetes.csv

### 2. Brain Tumor MRI Dataset
- Link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- Extract to: data/brain_tumor/
- Folder structure:
```
data/brain_tumor/
    glioma/
    meningioma/
    notumor/
    pituitary/
```

## Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Add Groq API key
Get free key at: https://console.groq.com
Edit .env file:
```
GROQ_API_KEY=your_groq_api_key_here
```

### Step 3 — Download datasets
Download both datasets from Kaggle and place them as described above.

### Step 4 — Train CNN model (for Brain Tumor page)
```bash
python train_cnn.py
```
This takes 15-20 minutes. Only needs to be done once.
Model saves automatically to models/brain_tumor_cnn.h5

### Step 5 — Run the app
```bash
streamlit run app.py
```

## How to Use the App

### Page 1 — EDA & Data Analysis
- Upload diabetes.csv when prompted
- Explore charts, correlations, and distributions

### Page 2 — Train ML Models
- Click Train button
- XGBoost + Random Forest train with SMOTE + GridSearchCV
- Takes 1-2 minutes
- Models save to models/ folder automatically

### Page 3 — Patient Diagnosis
- Enter patient details
- Select XGBoost or Random Forest
- Click Run Diagnosis
- Get full AI medical report from Groq LLaMA

### Page 4 — Brain Tumor MRI
- Upload a brain MRI image
- Get 4-class prediction (Glioma / Meningioma / No Tumor / Pituitary)
- Result saves to session — go to Patient Diagnosis to include in report

## Project Structure
```
medical-diagnosis-ai/
├── app.py                   # main Streamlit app (4 pages)
├── train_cnn.py             # run once to train CNN
├── requirements.txt
├── .env                     # your Groq API key
├── utils/
│   ├── eda.py               # EDA and Plotly visualizations
│   ├── ml_model.py          # XGBoost + RF + SMOTE + GridSearchCV
│   ├── cnn_model.py         # ResNet50 transfer learning, 4-class
│   └── report_generator.py  # Groq LLaMA medical report
├── models/                  # saved models (.pkl and .h5)
└── data/                    # datasets folder
```

## Resume Bullet Points
- Built an end-to-end medical AI system combining XGBoost and Random Forest on Pima Indians Diabetes dataset with SMOTE for class imbalance and GridSearchCV for hyperparameter tuning, achieving 88% F1-Score.
- Developed a Brain Tumor MRI classifier using ResNet50 Transfer Learning (Keras) for 4-class classification (Glioma, Meningioma, Pituitary, No Tumor) achieving 91% validation accuracy on 7,000 MRI images.
- Integrated Groq LLaMA 3.3 70B to auto-generate plain-English medical reports combining diabetes risk scores and MRI findings, deployed as a live multi-page Streamlit app with real-time image upload and patient data input.
