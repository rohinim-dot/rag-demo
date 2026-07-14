# RAG Mini-Demo (Retrieval-Augmented Generation)

A small, runnable demonstration of the **RAG** pattern: retrieving
relevant text from a knowledge base and using it to answer a question,
instead of relying purely on what a language model memorized during
training.

## Why I built this

RAG, embeddings, and vector search are called out specifically in
Junior ML Engineer job postings. I wanted to actually build a minimal
version of the pattern rather than just describe it — this project
retrieves real answers from a small AWS knowledge base I wrote myself.

## How RAG works (and what this demo does/doesn't include)

A full production RAG system has two stages:

1. **Retrieval** — convert documents and the user's question into
   vectors, then find the most similar documents (usually using a
   vector database like Pinecone, Chroma, or FAISS, with embeddings
   from a model like OpenAI's `text-embedding-3` or
   Sentence-Transformers).
2. **Generation** — send the retrieved text + the question to an LLM
   (e.g. Claude, GPT-4), which writes a grounded natural-language
   answer using that context.

**This demo fully implements retrieval**, using TF-IDF vectors and
cosine similarity instead of neural embeddings — this keeps it
runnable with zero API keys and no model downloads. **Generation is
stubbed**: instead of calling an LLM, it just returns the top
retrieved passage directly. The exact code for wiring in a real LLM
call (Anthropic's Claude API) is included as a comment in
`rag_pipeline.py`'s `generate_answer()` method.

## Running it

```bash
pip install -r requirements.txt
python rag_pipeline.py                          # runs 3 built-in test questions
python ask.py "What is Amazon S3 used for?"      # ask your own question
```

## A real limitation I found while testing this (and why it matters)

When I asked **"What is EC2 used for?"**, the system retrieved the S3
and Security Groups documents *ahead of* the actual EC2 document —
even though the question is clearly about EC2. Here's why: TF-IDF
matches on shared *words*, not shared *meaning*. The word "used"
appears in the S3 document's text, and "EC2" appears in the Security
Groups document, so both scored higher than the true EC2 document
even though neither is the best conceptual answer.

This is a well-known weakness of TF-IDF compared to real embeddings
(like Sentence-Transformers or OpenAI embeddings), which represent
words based on learned *meaning* rather than literal overlap. A
production RAG system would use real embeddings specifically to avoid
this failure mode.

## What I'd improve next

- Swap TF-IDF for real embeddings (`sentence-transformers`, e.g. the
  `all-MiniLM-L6-v2` model) and compare retrieval quality on the same
  test questions above
- Replace the stubbed `generate_answer()` with a real Claude or GPT-4
  API call so answers are written in natural language instead of
  returned as raw passages
- Swap the in-memory list for a real vector database (Chroma or FAISS)
  once the knowledge base grows beyond a handful of documents
- Add chunking logic for longer source documents (splitting large docs
  into smaller retrievable pieces)

## What I learned

- The difference between retrieval and generation as two distinct
  stages of a RAG pipeline
- Why TF-IDF is a reasonable retrieval baseline but fails on
  semantic (meaning-based) similarity — and saw a concrete example of
  it failing on my own test data
- How cosine similarity is used to rank documents by relevance to a
  query
