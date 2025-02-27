from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from ingest import create_cloud_sql_database_connection, get_embeddings, get_vector_store
from retrieve import get_relevant_documents, format_relevant_documents
from config import TABLE_NAME
import logging
import requests

load_dotenv()
app = FastAPI()
ENGINE = create_cloud_sql_database_connection()
EMBEDDING = get_embeddings()
logging.basicConfig(level=logging.INFO)

class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict

class UserInput(BaseModel):
    question: str
    language: str
    documents: List[DocumentResponse] = None

@app.post("/get_sources", response_model=List[DocumentResponse])
def get_sources(user_input: UserInput) -> List[DocumentResponse]:
    """
    Récupère les documents les plus pertinents pour une requête donnée.
    """
    vector_store = get_vector_store(ENGINE, TABLE_NAME, EMBEDDING)
    relevant_docs = get_relevant_documents(user_input.question, vector_store, 0.7)  # Ajout d'un seuil de similarité
    
    if not relevant_docs:
        return []
    
    result = []
    for doc in relevant_docs:
        # Créer une copie du dictionnaire des métadonnées
        metadata_with_score = dict(doc.metadata)
        
        # Assurer que le score est présent dans les métadonnées
        if 'score' in doc.metadata:
            metadata_with_score['similarity_score'] = float(doc.metadata['score'])
        elif hasattr(doc, 'metadata') and 'score' in doc.metadata:
            metadata_with_score['similarity_score'] = float(doc.metadata['score'])
        
        # Créer le DocumentResponse avec les métadonnées enrichies
        result.append(DocumentResponse(
            page_content=doc.page_content,
            metadata=metadata_with_score
        ))
    
    return result

@app.post("/answer")
def answer(user_input: UserInput):
    try:
        documents = user_input.documents

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.2,
            max_retries=2,
        )

        greeting_terms = ["hello", "hi", "hey", "bonjour", "salut", "coucou", "ça va", "comment vas-tu", 
                         "merci", "thank you", "thanks", "au revoir", "bye"]
        is_greeting = any(term in user_input.question.lower() for term in greeting_terms)

        # Formatter les documents uniquement si ce n’est pas une salutation et qu’on en a besoin
        formatted_docs = (
            format_relevant_documents(documents)
            if user_input.documents and not is_greeting
            else "No relevant documents provided."
        )

        # Vérifier la pertinence des documents (seuil minimal de similarité)
        min_similarity_threshold = 0.75  # Ajuste ce seuil selon tes besoins
        has_relevant_docs = False
        if documents and not is_greeting:
            for doc in documents:
                score = doc.metadata.get("similarity_score", 0.0)
                if score >= min_similarity_threshold:
                    has_relevant_docs = True
                    break

        if is_greeting:
            # Prompt pour les salutations
            prompt = ChatPromptTemplate.from_messages(
                messages=[
                    (
                        "system",
                        """You are AIHealthBot, a friendly medical assistant chatbot.
                        The user is just saying hello or a simple greeting. Respond in a friendly, conversational way.
                        Introduce yourself as a medical assistant and invite them to ask medical questions.
                        IMPORTANT: Your response must NOT include any of these elements:
                        - Do NOT include "Based on my general knowledge"
                        - Do NOT include "Detailed Answer from Medical Database"
                        - Do NOT include any document information or sources
                        - Do NOT provide ANY medical information
                        Just provide a simple, friendly greeting.
                        Respond in {language}.
                        """,
                    ),
                    ("human", "{question}"),
                ]
            )
            chain = prompt | llm
            response = chain.invoke({
                "language": user_input.language,
                "question": user_input.question
            }).content
        elif not has_relevant_docs:
            # Prompt pour les questions non médicales ou sans documents pertinents
            prompt = ChatPromptTemplate.from_messages(
                messages=[
                    (
                        "system",
                        """You are AIHealthBot, a medical assistant chatbot designed to answer medical questions.
                        The user has asked a question that is either not medical or for which no relevant medical information is available in the database.
                        Provide a detailed and informative answer to the question based on your general knowledge, including key details and useful information if applicable.
                        At the end, append this exact sentence:
                        "As a medical assistant, I’m here to help with health-related questions. Feel free to ask me anything about medical topics!"
                        
                        CRITICAL INSTRUCTIONS:
                        - Do NOT include "Detailed Answer from Medical Database" under any circumstances
                        - Do NOT mention or use any document information or sources, even if they were provided
                        - Focus solely on providing a complete and accurate answer to the question asked
                        - Respond in {language}.
                        """,
                    ),
                    ("human", "{question}"),
                ]
            )
            chain = prompt | llm
            response = chain.invoke({
                "language": user_input.language,
                "question": user_input.question
            }).content  # Pas de formatted_docs ici
        else:
            # Prompt standard pour les questions médicales avec documents pertinents
            prompt = ChatPromptTemplate.from_messages(
                messages=[
                    (
                        "system",
                        """CONTEXT:
                        The following DOCUMENTS contain relevant information that you MUST use to answer the question.
                        
                        DOCUMENTS:
                        {formatted_docs}
                        
                        INSTRUCTIONS:
                        1. First, provide a BRIEF general knowledge answer starting with "Based on my general knowledge," (limit to 2-3 sentences)
                        2. Then, present each document's information SEPARATELY without merging them.
                        3. Format your answer exactly like this:

                        Based on my general knowledge, [brief 2-3 sentence general answer].
                        
                        **Detailed Answer from Medical Database:**
                        
                        **Information 1:**
                        [Information from document 1, verbatim or slightly paraphrased]
                        Source: [source name] | Focus Area: [focus area] | Similarity Score: [similarity_score]
                        
                        **Information 2:**
                        [Information from document 2, verbatim or slightly paraphrased]
                        Source: [source name] | Focus Area: [focus area] | Similarity Score: [similarity_score]
                        
                        4. DO NOT merge information from different documents into a single paragraph.
                        5. Present each document's content SEPARATELY with its own heading and source information.
                        6. Keep each document's information in its original form without splitting it into separate sentences.
                        7. Answer in {language}.
                        
                        QUESTION:
                        {question}""",
                    ),
                    ("human", "The query is: {question}"),
                ]
            )
            chain = prompt | llm
            response = chain.invoke({
                "language": user_input.language,
                "question": user_input.question,
                "formatted_docs": formatted_docs
            }).content
        
        return {"message": response}
    except Exception as e:
        logging.error("Erreur lors du traitement de la requête : %s", e)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")