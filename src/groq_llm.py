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
- The document may use different section names — for example "Discussion" or "Summary" instead of "Conclusion", or "Results" instead of "Findings". Look for the meaning, not just the exact word.
- If asked about conclusions, also look at discussion, summary, findings, and results sections in the context
- If asked to summarize, give a thorough overview of all key points
- If the answer spans multiple points, explain each one clearly
- Only say "I couldn't find relevant information" if the topic is truly not mentioned anywhere in the context

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
        max_tokens=1024
    )

    return chat.choices[0].message.content