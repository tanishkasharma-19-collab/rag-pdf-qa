from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer_groq(query, docs):
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""You are a helpful assistant that answers questions based on the provided document context.

Instructions:
- Give a detailed and complete answer using the context below
- Follow any formatting instructions in the question (e.g. word limits, bullet points, summary)
- If the answer spans multiple points, explain each one clearly
- If the answer is not found in the context, say "I couldn't find relevant information in the document."
- Never make up information that isn't in the context

Context:
{context}

Question:
{query}

Answer:"""

    chat = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-8b-instant",
        max_tokens=1024  # increased from default to allow longer answers
    )

    return chat.choices[0].message.content