from pdf2image import convert_from_path
import pytesseract

def ocr_pdf(pdf_path, out_txt):
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=r"C:\\Users\\rprar\\Release-25.12.0-0\\poppler-25.12.0\\Library\\bin")
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page, lang="eng") + "\n"

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)

ocr_pdf(
    "data/raw_pdfs/Aryabhatiya v1.pdf",
    "data/extracted_text/Aryabhatiya v1.txt"
)
