from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_answer(query, docs):

    context = " ".join([d.page_content for d in docs])

    prompt = f"""
    Answer the question based on the context.

    Context:
    {context}

    Question:
    {query}
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer