import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


def generate_medical_report(patient_data: dict, ml_result: dict, tumor_result: dict = None):
    tumor_section = ""
    if tumor_result:
        tumor_section = f"""
Brain MRI Analysis (CNN - ResNet50 Transfer Learning):
- Finding: {tumor_result['label']}
- Confidence: {tumor_result['confidence']}%
- Description: {tumor_result['description']}
- All class probabilities: {tumor_result['all_probabilities']}
"""

    prompt = f"""You are a medical AI assistant. Based on the following patient data and AI model predictions, generate a clear, structured medical summary report in plain English.

Patient Information:
- Pregnancies: {patient_data.get('Pregnancies')}
- Glucose Level: {patient_data.get('Glucose')} mg/dL
- Blood Pressure: {patient_data.get('BloodPressure')} mmHg
- Skin Thickness: {patient_data.get('SkinThickness')} mm
- Insulin: {patient_data.get('Insulin')} mu U/ml
- BMI: {patient_data.get('BMI')}
- Diabetes Pedigree Function: {patient_data.get('DiabetesPedigreeFunction')}
- Age: {patient_data.get('Age')} years

Diabetes Risk Assessment (ML Model - {ml_result['model']}):
- Prediction: {ml_result['label']}
- Risk Probability: {ml_result['probability']}%

{tumor_section}

Generate a structured report with these sections:
1. Patient Summary
2. Key Risk Factors Identified
3. AI Model Findings (Diabetes + Brain MRI if available)
4. Recommended Next Steps

Keep it simple, clear, and easy for a non-medical person to understand.
End with a disclaimer that this is AI-generated and not a substitute for professional medical advice.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()
