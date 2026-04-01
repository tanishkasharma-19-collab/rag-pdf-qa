import fitz  # PyMuPDF
import base64
import os
from groq import Groq
from dotenv import load_dotenv
from langchain_core.documents import Document
from PIL import Image
import io

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def pdf_page_to_base64(page):
    """Render a PDF page to a base64-encoded PNG image."""
    mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode("utf-8")


def extract_text_with_groq_vision(base64_image):
    """Send a page image to Groq vision model and extract text."""
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Extract all the text from this image exactly as it appears. Only return the extracted text, nothing else."
                    }
                ]
            }
        ],
        max_tokens=2000
    )
    return response.choices[0].message.content.strip()


def load_pdf_with_ocr(path):
    """
    Extract text from a scanned/image-based PDF using Groq vision model.
    Renders each page as an image and sends it to Groq for OCR.
    Returns a list of LangChain Document objects (one per page).
    """
    docs = []

    try:
        pdf = fitz.open(path)

        for page_num in range(len(pdf)):
            page = pdf[page_num]

            # Render page to base64 image
            base64_image = pdf_page_to_base64(page)

            # Extract text using Groq vision
            text = extract_text_with_groq_vision(base64_image)

            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": path, "page": page_num + 1}
                ))

        pdf.close()
        print(f"Groq Vision OCR: extracted {len(docs)} pages from '{path}'")

    except Exception as e:
        print(f"Groq Vision OCR Error on '{path}': {e}")

    return docs


def load_image_text(path):
    """
    Extract text from a single image file using Groq vision model.
    """
    try:
        with open(path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        # Detect image type
        ext = path.split(".")[-1].lower()
        mime = "image/png" if ext == "png" else "image/jpeg"

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Extract all the text from this image exactly as it appears. Only return the extracted text, nothing else."
                        }
                    ]
                }
            ],
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Groq Vision OCR Error on image '{path}': {e}")
        return ""