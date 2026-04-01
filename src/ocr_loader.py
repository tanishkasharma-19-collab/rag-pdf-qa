import pytesseract
from PIL import Image
from langchain_core.documents import Document
import fitz  # PyMuPDF


def load_image_text(path):
    """
    Extract text from a single image file using Tesseract OCR.
    """
    try:
        image = Image.open(path)
        text = pytesseract.image_to_string(image)

        if not text.strip():
            return ""

        return text

    except Exception as e:
        print(f"OCR Error on image '{path}': {e}")
        return ""


def load_pdf_with_ocr(path):
    """
    Extract text from a scanned/image-based PDF using PyMuPDF + Tesseract OCR.
    Renders each PDF page as an image, then runs OCR on it.
    Returns a list of LangChain Document objects (one per page).
    """
    docs = []

    try:
        pdf = fitz.open(path)

        for page_num in range(len(pdf)):
            page = pdf[page_num]

            # Render page to image at 2x zoom for better OCR accuracy
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)

            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": path, "page": page_num + 1}
                ))

        pdf.close()

        print(f"OCR extracted {len(docs)} pages from '{path}'")

    except Exception as e:
        print(f"OCR PDF Error on '{path}': {e}")

    return docs