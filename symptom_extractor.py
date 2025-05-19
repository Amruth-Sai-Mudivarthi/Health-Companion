import re
from collections import Counter

def preprocess_text(text):
    """
    Preprocesses text by converting to lowercase, replacing underscores with spaces,
    and removing special characters.
    """
    # Convert to lowercase
    text = text.lower()
    # Replace underscores with spaces
    text = text.replace('_', ' ')
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def extract_symptoms(text, symptom_list):
    """
    Extracts symptoms from text by comparing with a predefined symptom list.
    
    Args:
        text (str): User input text describing symptoms
        symptom_list (list): List of all possible symptoms to match against
        
    Returns:
        list: Found symptoms from the symptom list
    """
    text = preprocess_text(text)
    found_symptoms = []
    
    # Check for each symptom in the list
    for symptom in symptom_list:
        processed_symptom = preprocess_text(symptom)
        if processed_symptom in text:
            found_symptoms.append(symptom)
    
    return found_symptoms