from PyPDF2 import PdfReader
import os

def load_pdf(file_path):
    """Extract text from PDF with improved handling"""
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"  
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    
    return text

def load_resume(uploaded_file):
    """Load and process a resume file"""
    if uploaded_file is not None:
        try:
            temp_path = os.path.join(os.getcwd(), "temp_resume.pdf")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            resume_text = load_pdf(temp_path)
            
            if not resume_text or len(resume_text.strip()) < 50:
                print("Warning: Very little text extracted from PDF. The file might be image-based.")
                
                
                try:
                    import pytesseract
                    from PIL import Image
                    from pdf2image import convert_from_path
                    
                    print("Attempting OCR on PDF...")
                    images = convert_from_path(temp_path)
                    ocr_text = ""
                    for img in images:
                        ocr_text += pytesseract.image_to_string(img) + "\n\n"
                    
                    if len(ocr_text.strip()) > 50:
                        resume_text = ocr_text
                        print("Successfully extracted text using OCR")
                except ImportError:
                    print("OCR libraries not available. Install pytesseract, PIL, and pdf2image.")
                except Exception as ocr_err:
                    print(f"OCR extraction failed: {ocr_err}")
            
            
            
            return resume_text
            
        except Exception as e:
            print(f"Error processing resume: {e}")
            return None
    return None