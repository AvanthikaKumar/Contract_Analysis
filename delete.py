# run this once from your project root
# python delete_index.py
 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
 
from ingestion.vector_store import vector_store
 
vector_store.delete_index()
print("Index deleted. Re-run your ingestion now.")