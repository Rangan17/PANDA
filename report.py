import json
from datetime import datetime

import requests

# Your Gemini API URL and key
GEMINI_API_URL = "........................................................................................"  # Replace with actual endpoint
GOOGLE_API_KEY = "........................................"  # Replace with your actual API key

# List of symptoms to classify from
SYMPTOMS_LIST = [
"Excessive worrying",  
"Feeling restless or on edge",  
"Irritability",  
"Difficulty concentrating",  
"Muscle tension",  
"Difficulty falling asleep or staying asleep",  
"Fatigue",  
"Racing thoughts",  
"Feeling overwhelmed",  
"Restlessness",  
"Feeling constantly on high alert",  
"Palpitations or rapid heart rate",  
"Shortness of breath",  
"Chest pain or discomfort",  
"Nausea or upset stomach",  
"Dizziness or lightheadedness",  
"Trembling or shaking",  
"Sweating excessively",  
"Feeling faint",  
"Dry mouth",  
"Difficulty swallowing",  
"Feeling hot or cold flashes",  
"Headaches",  
"Sadness",  
"Loss of interest or pleasure in activities",  
"Fatigue",  
"Sleep disturbances (insomnia or excessive sleep)",  
"Appetite changes (overeating or loss of appetite)",  
"Restlessness or feeling slowed down",  
"Difficulty concentrating or making decisions",  
"Feelings of guilt or worthlessness",  
"Irritability or anger",  
"Decreased energy levels",  
"Frequent crying or tearfulness",  
"Social withdrawal or isolation",  
"Feeling hopeless or pessimistic",  
"Thoughts of death or suicide",  
"Physical aches and pains with no apparent cause",  
"Changes in sexual desire or functioning",  
"Difficulty managing daily tasks or responsibilities",  
"Delusions",  
"Hallucinations (visual, auditory, tactile, olfactory)",  
"Disorganized thinking",  
"Disorganized speech",  
"Paranoia",  
"Incoherence",  
"Eccentric behavior",  
"Social withdrawal",  
"Lack of emotional expression",  
"Flat affect",  
"Catatonic behavior",  
"Bizarre or unusual beliefs",  
"Thought broadcasting (belief that one's thoughts are being projected to others)",  
"Thought insertion (belief that external forces are inserting thoughts into one's mind)",  
"Thought withdrawal (belief that thoughts are being taken away by external forces)",  
"Persecutory delusions (belief that one is being targeted, followed, or harmed)",  
"Chronic feelings of emptiness",  
"Impulsive and reckless behavior",  
"Difficulty forming and maintaining relationships",  
"Extreme mood swings",  
"Lack of empathy for others",  
"Intense fear of abandonment",  
"Excessive need for attention and validation",  
"Grandiose sense of self-importance",  
"Strong belief in one's own superiority",  
"Difficulty controlling anger and aggression",  
"Chronic feelings of sadness or hopelessness",  
"Inability to trust others",  
"Persistent feelings of anxiety or worry",  
"Extreme sensitivity to criticism or rejection",  
"Difficulty expressing emotions appropriately",  
"Tendency to manipulate or exploit others",  
"Unstable sense of self-image and identity",  
"Difficulty with social interactions",  
"Lack of eye contact",  
"Delayed language development",  
"Repetitive behaviors or movements",  
"Difficulty understanding nonverbal cues",  
"Inability to maintain friendships",  
"Sensitivity to sensory stimuli (e.g., loud noises, bright lights)",  
"Trouble with transitions or changes in routine",  
"Difficulty with impulse control",  
"Emotional outbursts or tantrums",  
"Fixation on certain objects or topics",  
"Developmental delays in motor skills",  
"Trouble with organization and planning",  
"Limited or unusual interests",  
"Difficulty with problem-solving or abstract thinking",  
"Uneven or atypical cognitive abilities",  
"Challenges with executive functioning",  
"Insomnia",  
"Excessive daytime sleepiness",  
"Irritability",  
"Difficulty falling asleep",  
"Waking up frequently during the night",  
"Snoring",  
"Difficulty staying asleep",  
"Restless leg syndrome",  
"Sleepwalking",  
"Night sweats",  
"Sleep apnea",  
"Morning headaches",  
"Difficulty concentrating",  
"Fatigue",  
"Forgetfulness",  
"Depression",  
"Anxiety",  
"Mood swings",  
"Decreased libido",  
"Weight gain",  
"Loud or irregular breathing during sleep",  
"Teeth grinding",  
"Dry mouth",  
"Frequent awakenings to urinate",  
"Flashbacks",  
"Nightmares",  
"Intrusive thoughts",  
"Avoidance of trauma-related stimuli",  
"Difficulty concentrating",  
"Trouble sleeping",  
"Irritability",  
"Hypervigilance",  
"Feeling on edge",  
"Jumpiness",  
"Startling easily",  
"Racing heartbeat",  
"Sweating",  
"Trembling or shaking",  
"Nausea or upset stomach",  
"Shortness of breath",  
"Dizziness",  
"Feeling disconnected from others",  
"Feeling numb or detached",  
"Feeling guilty or responsible for the trauma",  
"Experiencing a sense of doom or impending danger",  
"Memory problems",  
"Self-destructive behavior"
]

def extract_user_inputs(chat_history_file):
    """
    Extract user inputs from the chat history file.
    """
    user_inputs = []
    with open(chat_history_file, 'r') as file:
        for line in file:
            if line.startswith("You:"):
                # Extract the user input by removing 'You:'
                user_input = line.replace("You:", "").strip()
                user_inputs.append(user_input)
    return user_inputs

def classify_symptoms_with_gemini(user_input):
    """
    Call the Gemini API to classify symptoms from the user's input based on the predefined symptom list.
    """
    headers = {
        'Authorization': f'Bearer {GOOGLE_API_KEY}',
        'Content-Type': 'application/json'
    }

    # Send the user input and the predefined symptom list to the Gemini API
    payload = {
        'input_text': user_input,
        'symptoms_list': SYMPTOMS_LIST
    }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            classified_symptoms = result.get('symptoms', [])
            return classified_symptoms
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []

def generate_report(user_inputs, symptom_classifications):
    """
    Generate a mental health report based on the user inputs and symptom classifications.
    """
    # Get current time and date
    current_time = datetime.now().strftime("%H:%M")
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Initialize variables for report
    num_user_inputs = len(user_inputs)
    symptoms = set()  # Use a set to avoid duplicate symptoms

    # Collect all symptoms
    for symptom_list in symptom_classifications:
        symptoms.update(symptom_list)

    # Generate the report
    report = f"""
    Title: Your Mental Health Report from Dr.PANDA

    Time: {current_time}
    Date: {current_date}

    Number of user inputs: {num_user_inputs}

    Symptoms: {', '.join(symptoms)}

    Symptoms Definitions:
    """
    for symptom in symptoms:
        report += f"\n- {symptom}: Definition for {symptom}"

    report += "\n\nEnd of Report"

    return report

# Main execution
chat_history_file = "chat_history.txt"
user_inputs = extract_user_inputs(chat_history_file)

# Perform symptoms classification for each user input using Gemini API
symptom_classifications = []
for user_input in user_inputs:
    classified_symptoms = classify_symptoms_with_gemini(user_input)
    symptom_classifications.append(classified_symptoms)

# Generate the report
report = generate_report(user_inputs, symptom_classifications)

# Save the report to a file
with open("mental_health_report.txt", "w") as report_file:
    report_file.write(report)

print("Report generated and saved to 'mental_health_report.txt'.")
