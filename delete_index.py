"""
delete_index.py
---------------
One-time utility to delete and recreate the Azure AI Search index.
Run this whenever you need a clean slate — e.g. after changing
chunk IDs, vector dimensions, or index schema.
 
Usage:
    python delete_index.py
"""
 
import sys
from pathlib import Path
 
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
from ingestion.vector_store import vector_store
 
print("\n=== Azure AI Search Index Reset ===\n")
 
print("1. Deleting existing index...")
try:
    vector_store.delete_index()
    print("   Index deleted.\n")
except Exception as exc:
    print(f"   Index may not exist yet (safe to ignore): {exc}\n")
 
print("2. Recreating index with correct schema...")
try:
    vector_store.ensure_index_exists()
    print("   Index recreated successfully.\n")
except Exception as exc:
    print(f"   FAILED: {exc}\n")
    sys.exit(1)
 
print("Index reset complete. You can now re-upload your contracts.\n")