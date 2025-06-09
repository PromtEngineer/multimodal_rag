from typing import List, Tuple, Dict, Any
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, OcrMacOptions
from docling.datamodel.base_models import InputFormat

class PDFConverter:
    """
    A class to convert PDF files to structured Markdown using the docling library.
    """
    def __init__(self):
        """Initializes the docling document converter with forced OCR enabled for macOS."""
        try:
            # Configure the pipeline to force full-page OCR on macOS
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            
            ocr_options = OcrMacOptions(force_full_page_ocr=True)
            pipeline_options.ocr_options = ocr_options

            # Wrap the options in a format-specific configuration
            format_config = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
            
            self.converter = DocumentConverter(format_options=format_config)
            print("docling DocumentConverter initialized successfully with forced OCR on macOS.")
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
            
        print(f"Converting {pdf_path} to Markdown using docling (OCR enabled)...")
        pages_data = []
        try:
            result = self.converter.convert(pdf_path)
            markdown_content = result.document.export_to_markdown()
            
            metadata = {"source": pdf_path}
            pages_data.append((markdown_content, metadata))
            print(f"Successfully converted {pdf_path} with docling.")
            return pages_data
        except Exception as e:
            print(f"Error processing PDF {pdf_path} with docling: {e}")
            return []
