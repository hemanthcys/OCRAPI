from fastapi import FastAPI, File, UploadFile, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import openai
import io
import os

# Optional: Set path to Tesseract on Windows
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# ✅ Allow CORS for all domains (open access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow any origin
    allow_credentials=False,    # Must be False when using '*'
    allow_methods=["*"],
    allow_headers=["*"],
)

# OCR function using Tesseract
def extract_text_from_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    ocr_result = pytesseract.image_to_string(image)
    return ocr_result

# Call OpenAI to extract structured receipt info
def query_openai_for_extraction(ocr_text, openai_api_key: str = Header(...)):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """
You are an intelligent system that extracts sustainability and recycling data from shopping receipts.

Your job is to:
- Extract Product Name
- Identify Packaging Type (e.g., PET bottle, Tetra Pak, paper)
- Determine if packaging is recyclable
- Include local recycling instructions if location is available
- Present everything in this format:

[
    {
        "Product Name": "...",
        "Product Category": "...",
        "Material Composition": "...",
        "Packaging Type": "...",
        "Recyclable": "Yes/No/Conditional",
        "Recycling Instructions": "...",
        "Notes": "..."
    }
]
                    """
                },
                {"role": "user", "content": ocr_text}
            ],
            headers={"Authorization": f"Bearer {openai_api_key}"}
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.AuthenticationError as e:
        raise Exception(f"OpenAI Authentication Error: {e}")
    except openai.error.OpenAIError as e:
        raise Exception(f"Error communicating with OpenAI: {e}")

# API route to process the uploaded receipt image
@app.post("/ocr/")
async def process_receipt(file: UploadFile = File(...), openai_api_key: str = Header(...)):
    try:
        contents = await file.read()
        ocr_text = extract_text_from_image(contents)
        extracted_info = query_openai_for_extraction(ocr_text, openai_api_key)

        return JSONResponse(content={
            "ocr_text": ocr_text,
            "structured_data": extracted_info
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
