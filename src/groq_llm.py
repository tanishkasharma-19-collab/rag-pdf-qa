from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer_groq(query, docs):
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
Follow any instructions in the question exactly (e.g. word limits, bullet points, summary format).
If the answer is not found in the context, say "I couldn't find relevant information in the document."

Context:
{context}

Question:
{query}

Answer:"""

    chat = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-8b-instant"
    )

    return chat.choices[0].message.content