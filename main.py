# main.py (Azure OpenAI version)
import io, os, json
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document as DocxDocument
from openai import AzureOpenAI
import dotenv
dotenv.load_dotenv()
app = FastAPI(title="Resume Extract (Azure GPT-4o JSON)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION", "2024-06-01"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
MODEL = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")  # deployment name

class Project(BaseModel):
    name: str
    description: Optional[str] = None
    tech: List[str] = []
    role: Optional[str] = None
    impact: Optional[str] = None

class Experience(BaseModel):
    company: Optional[str] = None
    role: Optional[str] = None
    duration: Optional[str] = None
    responsibilities: List[str] = []
    tech: List[str] = []

class Education(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None
    details: Optional[str] = None

class ResumeExtraction(BaseModel):
    summary: Optional[str] = None
    skills: List[str] = []
    projects: List[Project] = []
    experience: List[Experience] = []
    education: List[Education] = []
    certifications: List[str] = []
    keywords: List[str] = []
    inferred_roles: List[str] = []

def read_pdf_bytes(b: bytes) -> str:
    with io.BytesIO(b) as fh:
        return pdf_extract_text(fh) or ""

def read_docx_bytes(b: bytes) -> str:
    with io.BytesIO(b) as fh:
        doc = DocxDocument(fh)
        return "\n".join(p.text for p in doc.paragraphs)

@app.post("/extract-skills", response_model=ResumeExtraction)
async def extract_skills(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if file.filename.lower().endswith(".pdf"):
            text = read_pdf_bytes(data)
        elif file.filename.lower().endswith(".docx"):
            text = read_docx_bytes(data)
        else:
            raise HTTPException(400, "Only PDF or DOCX supported")
        if not text or len(text.strip()) < 50:
            raise HTTPException(422, "Could not read meaningful text from the resume")

        # Force JSON output using response_format
        system_msg = (
            "You extract resume data and return ONLY valid JSON. "
            "No markdown, no commentary."
        )
        user_msg = f"""
Return a JSON object with these fields only:
- summary: string
- skills: string[]
- projects: {{ name: string, description?: string, tech?: string[], role?: string, impact?: string }}[]
- experience: {{ company?: string, role?: string, duration?: string, responsibilities?: string[], tech?: string[] }}[]
- education: {{ degree?: string, institution?: string, year?: string, details?: string }}[]
- certifications: string[]
- keywords: string[]
- inferred_roles: string[]

Rules:
- Be concise, deduplicate arrays, infer missing fields reasonably.
- If a section is absent, return an empty array for it.

RESUME TEXT:
{text}
""".strip()

        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        content = resp.choices[0].message.content
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as e:
            raise HTTPException(502, f"Model did not return valid JSON: {e}")

        # Validate/coerce with Pydantic
        try:
            return ResumeExtraction(**payload)
        except ValidationError:
            for k in ["skills","projects","experience","education","certifications","keywords","inferred_roles"]:
                payload.setdefault(k, [])
            return ResumeExtraction(**payload)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Extraction failed: {e}")