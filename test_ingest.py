import sys
sys.path.insert(0, r'D:\PrivateAI-Brain\normal-brain\privateai-v2\src')
from ingest import ingest_directory

docs = ingest_directory(r'D:\PrivateAI-Brain\my_knowledge')
print(f'Total chunks: {len(docs)}')
for d in docs[:5]:
    print(f'  {d["source"]}: {d["text"][:80]}...')
