import pandas as pd
import random
import requests
import time
import json
from tqdm import tqdm
import giskard as gs
import logging
from dotenv import load_dotenv
import os
import litellm

# Charger les variables d’environnement depuis .env
load_dotenv()

# Configuration
API_URL = "http://localhost:8181"
NUM_SAMPLES = 10  
SEED = 42
DATASET_PATH = "downloaded_files/medquad.csv"

logging.basicConfig(level=logging.INFO)

def load_dataset(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        if 'question' not in df.columns or 'answer' not in df.columns:
            raise ValueError("Le CSV doit contenir 'question' et 'answer'.")
        logging.info(f"Dataset chargé avec succès: {len(df)} entrées")
        return df
    except Exception as e:
        logging.error(f"Erreur lors du chargement du dataset: {e}")
        exit(1)

def select_random_samples(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    if len(df) < n:
        logging.warning(f"Le dataset contient moins de {n} exemples.")
        return df
    return df.sample(n)

class AIHealthBotModel:
    def __init__(self, api_url: str):
        self.api_url = api_url
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Évaluation des questions"):
            question = row['question']
            expected_answer = row.get('answer', '')
            start_time = time.time()
            
            try:
                sources_response = requests.post(
                    f"{self.api_url}/get_sources",
                    json={"question": question, "language": "en"},
                    timeout=30
                )
                documents = sources_response.json() if sources_response.status_code == 200 else []
                
                response = requests.post(
                    f"{self.api_url}/answer",
                    json={"question": question, "language": "en", "documents": documents},
                    timeout=30
                )
                answer = response.json().get("message", "") if response.status_code == 200 else f"Erreur: {response.status_code}"
                
            except Exception as e:
                answer = f"Erreur: {str(e)}"
                documents = []
            
            end_time = time.time()
            
            has_detailed_answer = "Detailed Answer from Medical Database" in answer
            avg_similarity_score = sum(doc["metadata"].get("similarity_score", 0.0) for doc in documents) / len(documents) if documents else 0.0
            expected_words = set(expected_answer.lower().split())
            answer_words = set(answer.lower().split())
            common_keywords = expected_words.intersection(answer_words)
            keyword_ratio = len(common_keywords) / len(expected_words) if expected_words else 0
            
            results.append({
                'question': question,
                'expected_answer': expected_answer,
                'response': answer,
                'response_time': end_time - start_time,
                'sources_count': len(documents),
                'avg_similarity_score': avg_similarity_score,
                'has_detailed_answer': has_detailed_answer,
                'keyword_ratio': keyword_ratio
            })
        
        return pd.DataFrame(results)
    
    def predict_for_giskard(self, df: pd.DataFrame) -> pd.Series:
        results = []
        for _, row in df.iterrows():
            question = row['question']
            try:
                sources_response = requests.post(f"{self.api_url}/get_sources", json={"question": question, "language": "en"}, timeout=30)
                documents = sources_response.json() if sources_response.status_code == 200 else []
                response = requests.post(f"{self.api_url}/answer", json={"question": question, "language": "en", "documents": documents}, timeout=30)
                answer = response.json().get("message", "") if response.status_code == 200 else f"Erreur: {response.status_code}"
            except Exception as e:
                answer = f"Erreur: {str(e)}"
            results.append(answer)
        return pd.Series(results)

def main():
    logging.info(f"Évaluation du chatbot AIHealthBot sur {NUM_SAMPLES} échantillons...")
    
    # Charger le dataset
    df = load_dataset(DATASET_PATH)
    samples = select_random_samples(df, NUM_SAMPLES, SEED)
    logging.info(f"Échantillons sélectionnés: {len(samples)}")
    
    # Créer le wrapper pour AIHealthBot
    aihealth_bot = AIHealthBotModel(API_URL)
    
    # Calculer les métriques personnalisées
    predictions = aihealth_bot.predict(samples)
    
    avg_response_time = predictions['response_time'].mean()
    max_response_time = predictions['response_time'].max()
    avg_sources_count = predictions['sources_count'].mean()
    avg_similarity_score = predictions['avg_similarity_score'].mean()
    detailed_answer_ratio = predictions['has_detailed_answer'].mean()
    avg_keyword_ratio = predictions['keyword_ratio'].mean()
    
    report = {
        "performance": {
            "avg_response_time": float(avg_response_time),
            "max_response_time": float(max_response_time),
            "avg_sources_count": float(avg_sources_count),
            "avg_similarity_score": float(avg_similarity_score),
        },
        "quality": {
            "detailed_answer_ratio": float(detailed_answer_ratio),
            "avg_keyword_ratio": float(avg_keyword_ratio),
            "samples_count": int(len(samples)),
        }
    }
    
    detailed_results = []
    for _, row in predictions.iterrows():
        detailed_results.append({
            "question": row['question'],
            "expected_answer": row['expected_answer'],
            "response": row['response'],
            "response_time": float(row['response_time']),
            "sources_count": int(row['sources_count']),
            "avg_similarity_score": float(row['avg_similarity_score']),
            "has_detailed_answer": bool(row['has_detailed_answer']),
            "keyword_ratio": float(row['keyword_ratio'])
        })
    
    full_report = {"summary": report, "detailed_results": detailed_results}
    with open('evaluation_results_aihealthbot.json', 'w') as f:
        json.dump(full_report, f, indent=2)
    
    print("\n===== RÉSUMÉ DE L'ÉVALUATION =====")
    print(f"Temps de réponse moyen: {avg_response_time:.2f}s")
    print(f"Temps de réponse maximum: {max_response_time:.2f}s")
    print(f"Nombre moyen de sources: {avg_sources_count:.2f}")
    print(f"Score de similarité moyen des documents: {avg_similarity_score:.2f}")
    print(f"Ratio de réponses détaillées: {detailed_answer_ratio:.2f}")
    print(f"Ratio moyen de mots-clés communs: {avg_keyword_ratio:.2f}")
    print(f"Résultats détaillés sauvegardés dans: evaluation_results_aihealthbot.json.json")
    
    # Vérifier que GOOGLE_API_KEY est chargée
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("La clé GOOGLE_API_KEY n’a pas été trouvée dans .env")
    
    import litellm
    litellm.google_api_key = google_api_key
    
    litellm.model = "gemini/gemini-1.5-pro"   
    
    # Intégration Giskard
    giskard_model = gs.Model(
        lambda df: aihealth_bot.predict_for_giskard(df),
        model_type="text_generation",
        name="AIHealthBot",
        description="A medical chatbot providing answers based on Medsquad dataset",
        feature_names=['question']
    )
    
    giskard_dataset = gs.Dataset(
        samples,
        name="Medsquad Test Set",
        target="answer"
    )
    
    scan_results = gs.scan(giskard_model, giskard_dataset)
    scan_results.to_html("giskard_report_aihealthbot.html")
    logging.info("Rapport Giskard généré: giskard_report_aihealthbot.html")

if __name__ == "__main__":
    main()