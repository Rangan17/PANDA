import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
from fpdf import FPDF

# Google Gemini API configuration
genai.configure(api_key=".............................................")

# Predefined set of symptoms
SYMPTOMS =[
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

# Function to load document into a list of sections
def load_document(filepath):
    with open(filepath, 'r', encoding="ISO-8859-1") as file:
        document_content = file.read().split("\n\n")  # Split into sections
    return document_content

def generate_document_embeddings(sections):
  """
  Generates embeddings for document sections with size limitation handling.
  """
  model = 'models/embedding-001'
  section_embeddings = []
  for section in sections:
    chunks = chunk_text(section, max_length=5000)  # Adjust max_length as needed
    for chunk in chunks:
      embedding = genai.embed_content(model=model, content=chunk, task_type="retrieval_document")
      section_embeddings.append(embedding['embedding'])
  return section_embeddings

# Function to generate embeddings for a text query
def get_embeddings(text):
    model = 'models/embedding-001'
    embedding = genai.embed_content(model=model, content=text, task_type="retrieval_document")
    return embedding['embedding']

# Function to find the most relevant section(s) from the document based on the query
def get_relevant_section(user_query, document_sections, document_embeddings):
    query_embedding = get_embeddings(user_query)
    
    # Compute cosine similarity between query and document sections
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    
    # Get the index of the most relevant section(s)
    top_indices = np.argsort(similarities)[-3:][::-1]  # Top 3 relevant sections
    
    # Ensure we do not access out-of-range indices
    relevant_sections = []
    for i in top_indices:
        if i < len(document_sections):
            relevant_sections.append(document_sections[i])
    
    return relevant_sections

# Function to create the prompt for RAG
def make_rag_prompt(query, relevant_passage):
    relevant_passage = ' '.join(relevant_passage)
    prompt = (
        f"You are a psychologist named Dr.PANDA, please provide insights or advice. "
        f"Give response within 50 words strictly not more than that"
        #f"Respond in a point-wise manner so that your response is easy to understand for everyone. "
        #f"Keep a check on the user's mental state, and output the symptoms relatable with the user and the most probable mental health condition in a single word from your document. "
        f"Maintain a polite and conversational tone. If the passage is irrelevant, feel free to ignore it.\n\n"
        #f"QUESTION: '{query}'\n"
        #f"PASSAGE: '{relevant_passage}'\n\n"
        #f"ANSWER:"
    )
    return prompt

# Function to chunk text into smaller parts
def chunk_text(text, max_length=2000):
    """
    Splits text into chunks of a specified maximum length.
    """
    chunks = []
    while len(text) > max_length:
        split_index = text.rfind(' ', 0, max_length)  # Find the last space before max_length
        if split_index == -1:  # No space found, split at max_length
            split_index = max_length
        chunks.append(text[:split_index].strip())
        text = text[split_index:]  # Remainder of the text
    chunks.append(text.strip())  # Append any remaining text
    return chunks

# Function to generate a response using the RAG setup
def generate_response(user_prompt):
    model = genai.GenerativeModel('gemini-pro')

    # Chunk the input prompt if it's too large
    prompt_chunks = chunk_text(user_prompt)
    responses = []

    for chunk in prompt_chunks:
        try:
            answer = model.generate_content(chunk)
            responses.append(answer.text)
        except Exception as e:
            print("")

    return " ".join(responses)  # Combine responses from all chunks

# Function to classify user input into predefined symptoms using LLM
def classify_symptoms(user_input):
    symptom_prompt = (
        f"""Based on the following user input, classify the user's mental state into one of the following symptoms: "Excessive worrying","Feeling restless or on edge", "Irritability",  "Difficulty concentrating",  "Muscle tension",  "Difficulty falling asleep or staying asleep",  "Fatigue",  "Racing thoughts",  "Feeling overwhelmed",  "Restlessness",  "Feeling constantly on high alert",  "Palpitations or rapid heart rate",  "Shortness of breath",  "Chest pain or discomfort",  "Nausea or upset stomach",  "Dizziness or lightheadedness",  "Trembling or shaking",  "Sweating excessively",  "Feeling faint",  "Dry mouth",  "Difficulty swallowing",  "Feeling hot or cold flashes",  "Headaches",  "Sadness",  "Loss of interest or pleasure in activities",  "Fatigue",  "Sleep disturbances (insomnia or excessive sleep)",  "Appetite changes (overeating or loss of appetite)",  "Restlessness or feeling slowed down",  "Difficulty concentrating or making decisions",  "Feelings of guilt or worthlessness",  "Irritability or anger",  "Decreased energy levels",  "Frequent crying or tearfulness",  "Social withdrawal or isolation",  "Feeling hopeless or pessimistic",  "Thoughts of death or suicide",  "Physical aches and pains with no apparent cause",  "Changes in sexual desire or functioning",  "Difficulty managing daily tasks or responsibilities",  "Delusions",  "Hallucinations (visual, auditory, tactile, olfactory)",  "Disorganized thinking",  "Disorganized speech",  "Paranoia",  "Incoherence",  "Eccentric behavior",  "Social withdrawal",  "Lack of emotional expression",  "Flat affect",  "Catatonic behavior",  "Bizarre or unusual beliefs",  "Thought broadcasting (belief that one thoughts are being projected to others)",  "Thought insertion (belief that external forces are inserting thoughts into one's mind)",  "Thought withdrawal (belief that thoughts are being taken away by external forces)",  "Persecutory delusions (belief that one is being targeted, followed, or harmed)",  "Chronic feelings of emptiness",  "Impulsive and reckless behavior",  "Difficulty forming and maintaining relationships",  "Extreme mood swings",  "Lack of empathy for others",  "Intense fear of abandonment",  "Excessive need for attention and validation",  "Grandiose sense of self-importance",  "Strong belief in one's own superiority",  "Difficulty controlling anger and aggression",  "Chronic feelings of sadness or hopelessness",  "Inability to trust others",  "Persistent feelings of anxiety or worry",  "Extreme sensitivity to criticism or rejection",  "Difficulty expressing emotions appropriately",  "Tendency to manipulate or exploit others",  "Unstable sense of self-image and identity",  "Difficulty with social interactions",  "Lack of eye contact",  "Delayed language development",  "Repetitive behaviors or movements",  "Difficulty understanding nonverbal cues",  "Inability to maintain friendships",  "Sensitivity to sensory stimuli (e.g., loud noises, bright lights)",  "Trouble with transitions or changes in routine",  "Difficulty with impulse control",  "Emotional outbursts or tantrums",  "Fixation on certain objects or topics",  "Developmental delays in motor skills",  "Trouble with organization and planning",  "Limited or unusual interests",  "Difficulty with problem-solving or abstract thinking",  "Uneven or atypical cognitive abilities",  "Challenges with executive functioning",  "Insomnia",  "Excessive daytime sleepiness",  "Irritability",  "Difficulty falling asleep",  "Waking up frequently during the night",  "Snoring",  "Difficulty staying asleep",  "Restless leg syndrome",  "Sleepwalking",  "Night sweats", "Sleep apnea",  "Morning headaches",  "Difficulty concentrating",  "Fatigue",  "Forgetfulness",  "Depression",  "Anxiety",  "Mood swings",  "Decreased libido",  "Weight gain",  "Loud or irregular breathing during sleep",  "Teeth grinding",  "Dry mouth",  "Frequent awakenings to urinate",  "Flashbacks",  "Nightmares",  "Intrusive thoughts",  "Avoidance of trauma-related stimuli",  "Difficulty concentrating",  "Trouble sleeping",  "Irritability",  "Hypervigilance",  "Feeling on edge",  "Jumpiness",  "Startling easily",  "Racing heartbeat",  "Sweating",  "Trembling or shaking",  "Nausea or upset stomach",  "Shortness of breath",  "Dizziness",  "Feeling disconnected from others",  "Feeling numb or detached",  "Feeling guilty or responsible for the trauma",  "Experiencing a sense of doom or impending danger",  "Memory problems",  "Self-destructive behavior"""
        #f"Respond with only the most likely symptom.\n\n"
        #f"USER INPUT: {user_input}\n\n"
        #f"SYMPTOM:"
    )
    
    model = genai.GenerativeModel('gemini-pro')
    classification = model.generate_content(symptom_prompt)
    # Convert the model output into a list of symptoms (ensure it splits and formats properly)
    classified_symptoms = [symptom.strip() for symptom in classification.text.strip().split(",") if symptom.strip() in SYMPTOMS]
    
    return classified_symptoms

# Function to generate a PDF mental health report
def generate_pdf_report(user_inputs, symptom_classifications):
    """
    Generate a PDF mental health report based on the user inputs and symptom classifications.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Your Mental Health Report from Dr.PANDA", ln=True, align='C')

    # Get current time and date
    current_time = datetime.now().strftime("%H:%M")
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Add Date and Time
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Time: {current_time}", ln=True)
    pdf.cell(200, 10, f"Date: {current_date}", ln=True)

    # Number of user inputs
    num_user_inputs = len(user_inputs)
    pdf.cell(200, 10, f"Number of user inputs: {num_user_inputs}", ln=True)

    # Collect and display all symptoms
    symptoms = set()  # Use a set to avoid duplicate symptoms
    for symptom_list in symptom_classifications:
        symptoms.update(symptom_list)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Symptoms:", ln=True)

    pdf.set_font("Arial", size=12)
    for symptom in symptoms:
        pdf.cell(200, 10, f"- {symptom}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Symptoms Definitions:", ln=True)
    pdf.set_font("Arial", size=12)
    for symptom in symptoms:
        pdf.cell(200, 10, f"- {symptom}: Definition for {symptom}", ln=True)

    # Save PDF
    pdf_file = "mental_health_report.pdf"
    pdf.output(pdf_file)
    print(f"PDF report generated and saved to '{pdf_file}'.")

# Main function to handle query, document retrieval, response generation, and symptom classification
def chatbot_response(query, document_path):
    # Load document and generate document embeddings
    document_sections = load_document(document_path)
    document_embeddings = generate_document_embeddings(document_sections)
    
    # Retrieve relevant document passage(s)
    relevant_text = get_relevant_section(query, document_sections, document_embeddings)
    
    # Create RAG prompt and generate response
    prompt = make_rag_prompt(query, relevant_passage=relevant_text)
    response = generate_response(prompt)
    
    # Classify user's mental state (based on input) into predefined symptoms
    symptom_classification = classify_symptoms(query)
    
    return response, symptom_classification

# Chatbot loop to continuously interact with the user and collect inputs for report generation
def chatbot_loop(document_path):
    # Initial prompt
    print("Welcome to Dr. PANDA's mental health chatbot!")
    #print("You can ask about your mental state or symptoms you're experiencing.")
    print("Type 'quit' or 'exit' to end the chat and generate a mental health report.")
    
    user_inputs = []
    symptom_classifications = []
    
    while True:
        # User input prompt
        #print("\nHow are you feeling today? You can also describe any symptoms or concerns.")
        user_query = input("You: ")
        
        if user_query.lower() in ["quit", "exit"]:
            print("Dr.PANDA: Thank you for the conversation. Take care of your mental health.")
            
            # Generate and save PDF report at the end of the conversation
            generate_pdf_report(user_inputs, symptom_classifications)
            print("Dr.PANDA: Your mental health report has been generated and saved.")
            break
        
        # Get chatbot's response and classified symptom
        response, classified_symptom = chatbot_response(user_query, document_path)
        
        # Save user input and symptom classification for report
        user_inputs.append(user_query)
        symptom_classifications.append(classified_symptom)
        
        # Display the chatbot's response and symptom classification
        print(f"\nDr.PANDA: {response}")
        #print(f"Chatbot (Symptom): It seems like you might be experiencing '{classified_symptom}'.")

# Example usage (assuming 'document.txt' contains relevant mental health content)
document_path = r"C:\Users\ASUS\Downloads\panda_report\psychological_symptoms.docx"
chatbot_loop(document_path)
