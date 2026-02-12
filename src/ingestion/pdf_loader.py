import fitz  # PyMuPDF
import re

def load_pdf_sections(path):
    """
    Loads a PDF and attempts to identify sections based on common header patterns.
    Returns a list of dicts: {"title": str, "content": str, "page": int}
    """
    doc = fitz.open(path)
    sections = []
    current_section = {"title": "Introduction", "content": "", "page": 1}
    
    # Common header pattern: e.g., "1. Introduction", "Abstract", "References"
    header_pattern = re.compile(r'^(\d+\.?\s+)?([A-Z][a-z]+(\s+[A-Z][a-z]+)*|[A-Z]{2,}(\s+[A-Z]{2,})*)$')

    for i, page in enumerate(doc):
        text = page.get_text("text")
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Simple heuristic: Short lines that match header patterns
            if len(line) < 60 and header_pattern.match(line):
                # Save previous section if it has content
                if current_section["content"].strip():
                    sections.append(current_section)
                
                current_section = {
                    "title": line,
                    "content": "",
                    "page": i + 1
                }
            else:
                current_section["content"] += line + "\n"
                
    # Append the last section
    if current_section["content"].strip():
        sections.append(current_section)
        
    return sections

if __name__ == "__main__":
    # Test with a local PDF if exists
    import sys
    if len(sys.argv) > 1:
        res = load_pdf_sections(sys.argv[1])
        for s in res:
            print(f"[{s['title']}] (Page {s['page']})")
            print(s['content'][:100] + "...")
            print("-" * 20)
