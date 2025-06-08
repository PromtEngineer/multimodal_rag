import pymupdf
from typing import List, Tuple, Dict, Any

class PDFConverter:
    """
    A class to convert PDF files and extract their text content page by page.
    """
    def convert_to_markdown(self, pdf_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Extracts text from each page of a PDF, preserving layout.
        """
        print(f"Extracting text from {pdf_path} using PyMuPDF...")
        pages_data = []
        try:
            with pymupdf.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    text_content = page.get_text("text")
                    metadata = {"page_number": page_num + 1}
                    pages_data.append((text_content, metadata))
            return pages_data
        except Exception as e:
            print(f"Error processing PDF {pdf_path} with PyMuPDF: {e}")
            return []
