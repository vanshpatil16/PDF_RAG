import re

SECTION_PATTERN = re.compile(
    r"\n(?P<title>(\d+\.?\s)?[A-Z][A-Za-z\s]{3,})\n"
)
def split_into_sections(pages):
    sections = []
    current_section = {"title": "Unknown", "content": ""}

    for page in pages:
        text = page["text"]
        matches = list(SECTION_PATTERN.finditer(text))

        if not matches:
            current_section["content"] += text
            continue

        last_idx = 0
        for match in matches:
            if current_section["content"]:
                sections.append(current_section)

            current_section = {
                "title": match.group("title").strip(),
                "content": text[match.end():],
                "page": page["page"]
            }
            last_idx = match.end()

    sections.append(current_section)
    return sections
