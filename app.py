import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow
from tensorflow.keras.models import load_model
from symptom_extractor import preprocess_text, extract_symptoms
import time
from PIL import Image
import base64

# Set page config
st.set_page_config(page_title="Health Companion", page_icon="🏥", layout="wide")

# Load the model and necessary objects
@st.cache_resource
def load_models():
    model = load_model('C:/Users/M Amruth Sai/Downloads/Smart Symptom Checker and Health Diary/disease_prediction_model.h5')
    with open('C:/Users/M Amruth Sai/Downloads/Smart Symptom Checker and Health Diary/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('C:/Users/M Amruth Sai/Downloads/Smart Symptom Checker and Health Diary/all_symptoms.pkl', 'rb') as f:
        all_symptoms = pickle.load(f)
    return model, label_encoder, all_symptoms

model, label_encoder, all_symptoms = load_models()

# Fitness assessment scoring system
FITNESS_CRITERIA = {
    'screen_time': {'low': 2, 'medium': 1, 'high': 0},
    'exercise': {'yes': 2, 'sometimes': 1, 'no': 0},
    'diet': {'excellent': 2, 'good': 1, 'poor': 0},
    'sleep': {'8+': 2, '6-8': 1, '<6': 0},
    'water': {'8+': 2, '4-7': 1, '<4': 0}
}

HEALTH_STATUS = {
    9: "Excellent! You're crushing it! 💪",
    7: "Good! Small tweaks could make it perfect! 👍",
    5: "Fair. Let's work on some improvements! ✨",
    3: "Needs work. Your health deserves attention! ❤️",
    0: "Critical. Please prioritize your health! �"
}

QUIZ_QUESTIONS = [
    {
        "question": "How often should you get a health check-up?",
        "options": ["Every 6 months", "Every year", "Every 2 years", "Only when sick"],
        "answer": 1
    },
    {
        "question": "Which of these is NOT a symptom of dehydration?",
        "options": ["Dark urine", "Headache", "Excessive sweating", "Clear urine"],
        "answer": 3
    },
    {
        "question": "What's the recommended daily water intake?",
        "options": ["1-2 glasses", "4-6 glasses", "8-10 glasses", "12+ glasses"],
        "answer": 2
    }
]

# Background image function
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover;
            background-attachment: fixed;
        }}
        .main {{
            background-color: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add background image (replace with your image path)
add_bg_from_local('C:/Users/M Amruth Sai/Downloads/Smart Symptom Checker and Health Diary/background.jpg')

# Function to predict disease
def predict_disease(symptoms):
    binary_vector = np.zeros(len(all_symptoms))
    for symptom in symptoms:
        if symptom in all_symptoms:
            idx = all_symptoms.index(symptom)
            binary_vector[idx] = 1
    
    binary_vector = binary_vector.reshape(1, -1)
    predictions = model.predict(binary_vector)
    predicted_class = np.argmax(predictions, axis=1)
    disease = label_encoder.inverse_transform(predicted_class)[0]
    confidence = np.max(predictions) * 100
    
    top3_indices = np.argsort(predictions[0])[-3:][::-1]
    top3_diseases = label_encoder.inverse_transform(top3_indices)
    top3_confidences = predictions[0][top3_indices] * 100
    
    return disease, confidence, list(zip(top3_diseases, top3_confidences))

# Fitness tracker section
def fitness_tracker():
    st.subheader("🏋️‍♂️ Fitness Tracker")
    st.write("Let's assess your daily health habits!")
    
    with st.form("fitness_form"):
        st.write("**Daily Habits Assessment**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            screen_time = st.radio("Screen time today:", 
                                  ["Low (<2 hours)", "Medium (2-6 hours)", "High (>6 hours)"],
                                  key='screen_time')
            
            exercise = st.radio("Did you exercise today?", 
                              ["Yes", "Sometimes (light activity)", "No"],
                              key='exercise')
            
        with col2:
            diet = st.radio("How was your diet today?",
                           ["Excellent (balanced meals)", "Good (some healthy choices)", "Poor (junk food)"],
                           key='diet')
            
            sleep = st.radio("How much sleep did you get last night?",
                           ["8+ hours", "6-8 hours", "Less than 6 hours"],
                           key='sleep')
            
            water = st.radio("Glasses of water consumed today:",
                           ["8+ glasses", "4-7 glasses", "Less than 4 glasses"],
                           key='water')
        
        submitted = st.form_submit_button("Calculate My Health Score")
        
        if submitted:
            # Calculate health score
            score = 0
            score += FITNESS_CRITERIA['screen_time'][screen_time.split(' ')[0].lower()]
            score += FITNESS_CRITERIA['exercise'][exercise.split(' ')[0].lower()]
            score += FITNESS_CRITERIA['diet'][diet.split(' ')[0].lower()]
            score += FITNESS_CRITERIA['sleep'][sleep.split(' ')[0].lower()]
            score += FITNESS_CRITERIA['water'][water.split(' ')[0].lower()]
            
            # Determine health status
            status = next(v for k, v in HEALTH_STATUS.items() if score >= k)
            
            # Display results with animation
            with st.spinner('Calculating your health score...'):
                time.sleep(1.5)
                
            st.success(f"**Your Health Score: {score}/10**")
            st.balloons()
            st.markdown(f"### {status}")
            
            # Show recommendations based on score
            if score >= 8:
                st.info("Recommendation: Keep up the great work! Consider adding meditation for mental health.")
            elif score >= 5:
                st.warning("Recommendation: Try adding 15 minutes of exercise and one more glass of water daily.")
            else:
                st.error("Recommendation: Prioritize sleep and reduce screen time. Small changes make big differences!")

# Health quiz section
def health_quiz():
    st.subheader("🧠 Health Knowledge Quiz")
    st.write("Test your health knowledge with this quick quiz!")
    
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = [None] * len(QUIZ_QUESTIONS)
        st.session_state.quiz_submitted = False
    
    for i, question in enumerate(QUIZ_QUESTIONS):
        st.write(f"**{i+1}. {question['question']}**")
        options = question["options"]
        
        # Display as radio buttons
        selected = st.radio(
            f"Q{i+1}", 
            options, 
            index=st.session_state.quiz_answers[i] if st.session_state.quiz_answers[i] is not None else 0,
            key=f"quiz_{i}"
        )
        st.session_state.quiz_answers[i] = options.index(selected)
    
    if st.button("Submit Quiz"):
        st.session_state.quiz_submitted = True
        score = sum(1 for i, q in enumerate(QUIZ_QUESTIONS) 
                   if st.session_state.quiz_answers[i] == q["answer"])
        
        st.success(f"**Your Score: {score}/{len(QUIZ_QUESTIONS)}**")
        
        if score == len(QUIZ_QUESTIONS):
            st.balloons()
            st.markdown("🎉 **Perfect! You're a health expert!**")
        elif score >= len(QUIZ_QUESTIONS)/2:
            st.markdown("👍 **Good job! You know quite a bit!**")
        else:
            st.markdown("💡 **Keep learning - your health will thank you!**")
        
        # Show correct answers
        st.write("**Correct Answers:**")
        for i, q in enumerate(QUIZ_QUESTIONS):
            st.write(f"{i+1}. {q['options'][q['answer']]}")

# Main app
def main():
    st.title("🏥 Health Companion")
    
    tab1, tab2, tab3 = st.tabs(["Symptom Checker", "Fitness Tracker", "Health Quiz"])
    
    with tab1:
        st.subheader("🔍 Symptom Checker")
        user_input = st.text_area("Describe your symptoms:", 
                                placeholder="e.g., I have been experiencing itching, skin rash, and fatigue...")
        
        if st.button("Analyze Symptoms"):
            if user_input:
                found_symptoms = extract_symptoms(user_input, all_symptoms)
                
                if found_symptoms:
                    st.subheader("Identified Symptoms:")
                    st.write(", ".join(found_symptoms))
                    
                    disease, confidence, top3 = predict_disease(found_symptoms)
                    
                    st.subheader("Analysis Results:")
                    
                    if confidence >= 75:
                        st.success(f"Most likely condition: **{disease}** (confidence: {confidence:.2f}%)")
                        st.write("Other possibilities:")
                        for d, c in top3[1:]:
                            st.write(f"- {d} ({c:.2f}%)")
                    else:
                        st.warning("No known symptoms were identified. Here are the top possibilities:")
                        for d, c in top3:
                            st.write(f"- {d} ({c:.2f}%)")
                        st.info("Consider consulting a healthcare professional for more accurate diagnosis.")
                    
                    st.subheader("Symptom Analysis:")
                    symptom_importance = {}
                    for symptom in found_symptoms:
                        temp_symptoms = [s for s in found_symptoms if s != symptom]
                        if temp_symptoms:
                            _, temp_confidence, _ = predict_disease(temp_symptoms)
                            importance = confidence - temp_confidence
                            symptom_importance[symptom] = importance
                    
                    sorted_importance = sorted(symptom_importance.items(), key=lambda x: x[1], reverse=True)
                    
                    for symptom, importance in sorted_importance:
                        st.write(f"- {symptom}: {'+' if importance >= 0 else ''}{importance:.2f}% impact")
                else:
                    st.warning("No recognizable symptoms found. Try describing differently with more details.")
            else:
                st.warning("Please describe your symptoms to get analysis.")
    
    with tab2:
        fitness_tracker()
    
    with tab3:
        health_quiz()

if __name__ == "__main__":
    main()