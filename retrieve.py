import os
from ingest import create_cloud_sql_database_connection, get_embeddings, get_vector_store
from langchain_google_cloud_sql_pg import PostgresVectorStore
from langchain_core.documents.base import Document
from config import TABLE_NAME

def get_relevant_documents(query: str, vector_store: PostgresVectorStore, similarity_threshold: float) -> list[Document]:
    """
    Retrieve relevant documents based on a query using a vector store.

    Args:
        query (str): The search query string.
        vector_store (PostgresVectorStore): An instance of PostgresVectorStore used to retrieve documents.

    Returns:
        list[Document]: A list of documents relevant to the query.
    """
    relevant_docs_scores = vector_store.similarity_search_with_relevance_scores(
        query=query, k=2)
    for doc, score in relevant_docs_scores:
        doc.metadata["score"] = score
    relevant_docs = [doc for doc, _ in relevant_docs_scores]
    return relevant_docs


def format_relevant_documents(documents: list[Document]) -> str:
    """
    Formate les documents pertinents sous forme de texte clair et bien structuré.
    """
    if not documents or len(documents) == 0:
        return "No relevant documents found."
        
    formatted_docs = []
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "Unknown source")
        answer = doc.metadata.get("answer", "No answer available")
        focus_area = doc.metadata.get("focus_area", "Unknown area")
        
        # Get similarity score from metadata or document attribute
        score = None
        if 'similarity_score' in doc.metadata:
            score = doc.metadata['similarity_score']
        elif hasattr(doc, 'score'):
            score = doc.score
            
        score_str = f", Similarity Score: {score:.4f}" if score is not None else ""
        
        formatted_docs.append(
            f"### DOCUMENT {i+1}\n\n"
            f"**Question:** {doc.page_content}\n\n"
            f"**Answer:** {answer}\n\n"
            f"**Source:** {source}\n\n"
            f"**Focus Area:** {focus_area}{score_str}\n"
        )
    return "\n\n".join(formatted_docs)

if __name__ == '__main__':
    engine = create_cloud_sql_database_connection()
    embedding = get_embeddings()
    vector_store = get_vector_store(engine, TABLE_NAME, embedding)
    documents = get_relevant_documents("large language models", vector_store)
    assert len(documents) > 0, "No documents found for the query"
    doc_str = format_relevant_documents(documents)
    assert len(doc_str) > 0, "No documents formatted successfully"
    print("All tests passed successfully.")
