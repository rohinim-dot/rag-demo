"""
RAG Mini-Demo — Retrieval-Augmented Generation Pipeline
---------------------------------------------------------
RAG is a pattern where, instead of asking an LLM to answer purely from
what it memorized during training, you first RETRIEVE relevant text
from your own knowledge base, then hand that text to the model as
context so it can GENERATE a grounded, up-to-date answer.

This demo has two stages:

  1. RETRIEVAL (fully working, runs locally, no API needed)
     - Converts each document and the user's question into numeric
       vectors and finds the most similar documents using cosine
       similarity. This stands in for a "vector database" — in
       production you'd typically use a real vector DB (e.g. Pinecone,
       Chroma, FAISS) with embeddings from a model like OpenAI's
       text-embedding or Sentence-Transformers, rather than TF-IDF.
       TF-IDF is used here because it requires no external model
       downloads and no API key, so this runs anywhere.

  2. GENERATION (stubbed — see generate_answer() below)
     - In a real system, this step sends the retrieved text + the
       question to an LLM (e.g. Claude, GPT-4) which writes a natural
       language answer grounded in that context. Since that needs a
       paid API key, this demo instead does simple extractive
       generation: it returns the most relevant retrieved passage(s)
       directly. The code comments show exactly where a real LLM call
       would go.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from documents import DOCUMENTS


class RAGPipeline:
    def __init__(self, documents):
        self.documents = documents
        self.texts = [doc["text"] for doc in documents]

        # Build the TF-IDF "vector index" over all documents once, up front.
        # This is the local stand-in for a vector database + embeddings.
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 2):
        """
        Retrieval step: find the top_k most relevant documents for a
        given query, using cosine similarity between TF-IDF vectors.
        """
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]

        # Rank documents by similarity score, highest first
        ranked_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            results.append({
                "id": self.documents[idx]["id"],
                "text": self.documents[idx]["text"],
                "score": round(float(similarities[idx]), 3),
            })
        return results

    def generate_answer(self, query: str, retrieved_docs: list) -> str:
        """
        Generation step (stubbed).

        In a real RAG system, this is where you'd call an LLM API,
        e.g.:

            import anthropic
            client = anthropic.Anthropic(api_key="...")
            context = "\\n\\n".join(d["text"] for d in retrieved_docs)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": f"Context:\\n{context}\\n\\nQuestion: {query}\\n"
                               f"Answer using only the context above."
                }]
            )
            return response.content[0].text

        Since that needs a paid API key, this demo instead does simple
        extractive generation: it just returns the top retrieved
        passage(s) directly, with no rewriting.
        """
        if not retrieved_docs:
            return "No relevant information found in the knowledge base."

        best_match = retrieved_docs[0]
        return (
            f"Based on the most relevant document ({best_match['id']}, "
            f"similarity score {best_match['score']}):\n\n{best_match['text']}"
        )

    def query(self, question: str, top_k: int = 2):
        """Full RAG pipeline: retrieve relevant docs, then generate an answer."""
        retrieved = self.retrieve(question, top_k=top_k)
        answer = self.generate_answer(question, retrieved)
        return {
            "question": question,
            "retrieved_docs": retrieved,
            "answer": answer,
        }


if __name__ == "__main__":
    rag = RAGPipeline(DOCUMENTS)

    test_questions = [
        "How do I control who can access my AWS resources?",
        "What service should I use to store files in the cloud?",
        "How can I monitor my cloud infrastructure for issues?",
    ]

    for question in test_questions:
        result = rag.query(question, top_k=2)
        print(f"Q: {result['question']}")
        print(f"\nRetrieved documents:")
        for doc in result["retrieved_docs"]:
            print(f"  - [{doc['id']}] (score={doc['score']}) {doc['text'][:80]}...")
        print(f"\nAnswer:\n{result['answer']}")
        print("\n" + "=" * 80 + "\n")
