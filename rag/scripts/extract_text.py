# scripts/extract_text.py
import pdfplumber
from pathlib import Path

pdf_path = Path("data/raw_pdfs/Brihat Samhita.pdf")
output_path = Path("data/extracted_text/Brihat Samhita.txt")

with pdfplumber.open(pdf_path) as pdf:
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)

output_path.write_text(text, encoding="utf-8")
