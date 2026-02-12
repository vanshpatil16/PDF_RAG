from src.generation.answer_generator import QwenAPIGenerator
from src.generation.prompt import (
    get_qwen_messages, 
    SECTION_AWARE_SYSTEM_PROMPT, 
    format_context_chunks
)

def test_section_aware_generation():
    generator = QwenAPIGenerator()
    
    # Mock context chunks
    mock_chunks = [
        {
            "section": "Methodology",
            "chunk_id": 2,
            "text": "The model uses a transformer-based encoder with self-attention layers."
        },
        {
            "section": "Results",
            "chunk_id": 1,
            "text": "The proposed approach improves accuracy by 12% over baselines."
        }
    ]
    
    formatted_context = format_context_chunks(mock_chunks)
    query = "What model architecture is used and how effective is it?"
    
    messages = get_qwen_messages(SECTION_AWARE_SYSTEM_PROMPT, query, formatted_context)
    
    print(f"Query: {query}")
    print("\nFormatted Context:")
    print("-" * 20)
    print(formatted_context)
    print("-" * 20)
    
    print("\nGenerating response...")
    try:
        response = generator.generate_response(messages)
        print("\nModel Output:")
        print("-" * 20)
        print(response)
        print("-" * 20)
        
        # Simple check for citations in output
        if "(Section:" in response and "Chunk:" in response:
            print("\nVerification: Citations found in output! ✅")
        else:
            print("\nVerification: Citations NOT found in output. ❌")
            
        print("\nSuccess!")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    test_section_aware_generation()
