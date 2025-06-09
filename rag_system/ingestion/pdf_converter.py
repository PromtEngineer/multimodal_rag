from typing import List, Tuple, Dict, Any
from docling.document_converter import DocumentConverter

class PDFConverter:
    """
    A class to convert PDF files to structured Markdown using the docling library.
    """
    def __init__(self):
        """Initializes the docling document converter."""
        try:
            self.converter = DocumentConverter()
            print("docling DocumentConverter initialized successfully.")
        except Exception as e:
            print(f"Error initializing docling DocumentConverter: {e}")
            self.converter = None

    def convert_to_markdown(self, pdf_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Converts a PDF to a single Markdown string, preserving layout and tables.
        """
        if not self.converter:
            print("docling converter not available. Skipping conversion.")
            return []
            
        print(f"Converting {pdf_path} to Markdown using docling...")
        pages_data = []
        try:
            # docling converts the entire document at once
            result = self.converter.convert(pdf_path)
            markdown_content = result.document.export_to_markdown()
            
            # Return as a single item list to match the pipeline's expected format
            metadata = {"source": pdf_path}
            pages_data.append((markdown_content, metadata))
            print(f"Successfully converted {pdf_path} with docling.")
            return pages_data
        except Exception as e:
            print(f"Error processing PDF {pdf_path} with docling: {e}")
            return []
