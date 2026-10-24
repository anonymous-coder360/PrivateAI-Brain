"""
Document ingestion utility for RAG system
Supports: txt, md, pdf, jsonl, csv
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import hashlib

try:
    import fitz  # PyMuPDF for PDF
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    import pandas as pd
    HAS_CSV = True
except ImportError:
    HAS_CSV = False


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:  # Skip tiny chunks
            chunks.append(chunk)

    return chunks


def load_text_file(path: Path) -> List[Dict[str, Any]]:
    """Load plain text or markdown file"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = chunk_text(content)
    return [
        {
            "text": chunk,
            "source": str(path),
            "metadata": {
                "file_type": path.suffix,
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
        }
        for i, chunk in enumerate(chunks)
    ]


def load_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file (one JSON per line)"""
    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                try:
                    data = json.loads(line)
                    # Handle different JSONL formats
                    if "text" in data:
                        text = data["text"]
                    elif "content" in data:
                        text = data["content"]
                    elif "messages" in data:
                        # Chat format - combine messages
                        text = " ".join(
                            m.get("content", "") for m in data["messages"]
                        )
                    else:
                        text = json.dumps(data)

                    chunks = chunk_text(text)
                    for j, chunk in enumerate(chunks):
                        documents.append({
                            "text": chunk,
                            "source": str(path),
                            "metadata": {
                                "file_type": ".jsonl",
                                "line_index": i,
                                "chunk_index": j,
                                "total_chunks": len(chunks),
                            },
                        })
                except json.JSONDecodeError:
                    continue

    return documents


def load_pdf_file(path: Path) -> List[Dict[str, Any]]:
    """Load PDF file"""
    if not HAS_PDF:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")

    documents = []
    doc = fitz.open(path)

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "source": str(path),
                    "metadata": {
                        "file_type": ".pdf",
                        "page": page_num + 1,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                })

    doc.close()
    return documents


def load_csv_file(path: Path, text_column: str = "text") -> List[Dict[str, Any]]:
    """Load CSV file"""
    if not HAS_CSV:
        raise RuntimeError("pandas not installed. Run: pip install pandas")

    documents = []
    df = pd.read_csv(path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Columns: {df.columns.tolist()}")

    for i, row in df.iterrows():
        text = str(row[text_column])
        chunks = chunk_text(text)
        for j, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "source": str(path),
                "metadata": {
                    "file_type": ".csv",
                    "row_index": i,
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                    **{k: str(v) for k, v in row.items() if k != text_column},
                },
            })

    return documents


def ingest_directory(
    directory: str,
    recursive: bool = True,
    extensions: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Ingest all supported files from a directory
    Returns list of document chunks ready for RAG
    """
    if extensions is None:
        extensions = [".txt", ".md", ".jsonl", ".pdf", ".csv"]

    directory = Path(directory)
    all_docs = []

    pattern = "**/*" if recursive else "*"

    for file_path in directory.glob(pattern):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in extensions:
            continue

        print(f"Processing: {file_path}")

        try:
            if file_path.suffix.lower() in [".txt", ".md"]:
                docs = load_text_file(file_path)
            elif file_path.suffix.lower() == ".jsonl":
                docs = load_jsonl_file(file_path)
            elif file_path.suffix.lower() == ".pdf":
                docs = load_pdf_file(file_path)
            elif file_path.suffix.lower() == ".csv":
                docs = load_csv_file(file_path)
            else:
                continue

            all_docs.extend(docs)
            print(f"  -> {len(docs)} chunks")

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nTotal documents: {len(all_docs)}")
    return all_docs


def main():
    """CLI for document ingestion"""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents for RAG")
    parser.add_argument("directory", help="Directory to ingest")
    parser.add_argument("--output", "-o", help="Output JSONL file")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursive")
    args = parser.parse_args()

    docs = ingest_directory(args.directory, recursive=args.recursive)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()