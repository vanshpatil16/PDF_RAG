from FlagEmbedding import FlagModel
import sys

try:
    print("Attempting to load BGE base model...")
    model = FlagModel('BAAI/bge-base-en-v1.5', use_fp16=True)
    print("Model loaded successfully!")
    emb = model.encode(["test"])
    print(f"Encoding successful! Shape: {emb.shape}")
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
