from typing import List, Dict

SECTION_AWARE_SYSTEM_PROMPT = """You are a section-aware research paper assistant.

You MUST answer questions using ONLY the provided context chunks.
Each chunk belongs to a specific section of the paper.

Rules:
1. Do NOT use any external knowledge.
2. If the answer is not present in the context, respond with:
   "The information is not available in the provided document."
3. Every factual statement MUST be cited.
4. Citations must include the section name and chunk id.
5. Do NOT mention chunks or sections that are not used.
6. Be concise, clear, and academic in tone.
7. Do NOT hallucinate or infer missing information.

Citation format:
(Section: <section_name>, Chunk: <chunk_id>)"""

def format_context_chunks(chunks: List[Dict]) -> str:
    """
    Formats a list of context chunks into a string for the model.
    Each chunk is expected to have 'section', 'chunk_id', and 'text' keys.
    """
    formatted_chunks = []
    for chunk in chunks:
        section = chunk.get("section", "Unknown")
        chunk_id = chunk.get("chunk_id", "Unknown")
        text = chunk.get("text", "")
        formatted_chunks.append(f"[Section: {section}, Chunk: {chunk_id}]\n{text}")
    
    return "\n\n".join(formatted_chunks)

def get_qwen_messages(system_prompt: str, user_query: str, context: str = None) -> list:
    """
    Constructs messages for Qwen model.
    """
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    if context:
        user_content = f"Context:\n{context}\n\nQuestion:\n{user_query}"
    else:
        user_content = user_query
        
    messages.append({"role": "user", "content": user_content})
    return messages

DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
