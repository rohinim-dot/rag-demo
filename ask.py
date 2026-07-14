"""
Interactive query script for the RAG demo.

Usage:
    python ask.py "your question here"
"""

import sys
from rag_pipeline import RAGPipeline
from documents import DOCUMENTS


def main():
    if len(sys.argv) < 2:
        print('Usage: python ask.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    rag = RAGPipeline(DOCUMENTS)
    result = rag.query(question, top_k=2)

    print(f"Question: {result['question']}\n")
    print("Top retrieved documents:")
    for doc in result["retrieved_docs"]:
        print(f"  - [{doc['id']}] (similarity={doc['score']}) {doc['text']}")
    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()
