import google.generativeai as genai
import clickhouse_connect
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
from os import name

# Google Gemini API configuration
genai.configure(api_key=".......................")

# Clickhouse/MyScaleDB connection
client = clickhouse_connect.get_client(
      host='msc-c7939b41.us-east-1.aws.myscale.com',
      port=443,
      username='rangans_org_default',
      password='passwd_xMKZKJUpE4jeUX'
  )

# Function to generate embeddings for a text query
def get_embeddings(text):
    model = 'models/embedding-001'
    embedding = genai.embed_content(model=model, content=text, task_type="retrieval_document")
    return embedding['embedding']

# Function to retrieve relevant documents based on user query
def get_relevant_docs(user_query):
    query_embeddings = get_embeddings(user_query)
    results = client.query(f"""
        SELECT page_content,
        distance(embeddings, {query_embeddings}) as dist FROM default.handbook ORDER BY dist LIMIT 3
    """)
    relevant_docs = [row['page_content'] for row in results.named_results()]
    return relevant_docs

# Function to create the prompt for RAG
def make_rag_prompt(query, relevant_passage):
    relevant_passage = ' '.join(relevant_passage)
    prompt = (
        f"You are a psychologist named Dr.PANDA, please provide insights or advice. "
        f"Respond in a point-wise manner so that your response is easy to understand for everyone. "
        f"Keep a check on the user's mental state, and output the symptoms relatable with the user and the most probable mental health condition in a single word from your database text. "
        f"Maintain a polite and conversational tone. If the passage is irrelevant, feel free to ignore it.\n\n"
        f"QUESTION: '{query}'\n"
        f"PASSAGE: '{relevant_passage}'\n\n"
        f"ANSWER:"
    )
    return prompt

# Function to generate a response using the RAG setup
def generate_response(user_prompt):
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(user_prompt)
    return answer.text

# Main function to handle query and response
def generate_answer(query):
    relevant_text = get_relevant_docs(query)
    prompt = make_rag_prompt(query, relevant_passage=relevant_text)
    answer = generate_response(prompt)
    return answer

# Function to save the answer in a PDF report
def save_report_to_pdf(query, answer):
    # Generate timestamp
    report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Define the PDF file name
    pdf_file_name = r"C:\Users\UDAY SANKAR\Downloads\RUSA_Panda\mental_health_report.pdf"

    # Create a new PDF file using ReportLab
    c = canvas.Canvas(pdf_file_name, pagesize=letter)

    # Set Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Mental Health Report")

    # Add timestamp
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f"Report Generated on: {report_time}")

    # Draw a box to contain the query and answer
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(1)
    c.rect(100, 500, 400, 200, stroke=1, fill=0)

    # Print the Query and Answer within the box
    c.setFont("Helvetica", 10)
    text = c.beginText(110, 680)
    text.setFont("Helvetica", 12)

    # Add query to the report
    text.textLine(f"Query: {query}")
    text.textLine("")

    # Add answer to the report (Split long answers to fit in the box)
    for line in answer.split("\n"):
        text.textLine(line)

    c.drawText(text)

    # Save and close the PDF
    c.save()
    print(f"PDF report saved as {pdf_file_name}")
# Example usage
if name == "main":
    query = "got no one to talk to have no one around i ve been procrastinating on something for so long and i have no idea when i ll ever become serious or steadfast i just feel like a total waste i ve isolated myself which is making me go crazy right now no friend at all i m literally alone now feel like shit. What is this according to you"
    answer = generate_answer(query)
    print(f"Answer: {answer}")
    
    # Save the answer to a PDF report
    save_report_to_pdf(query, answer)
