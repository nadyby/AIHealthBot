import streamlit as st
from PIL import Image
import requests
from datetime import datetime

# Page configuration
st.set_page_config(page_title="AIHealthBot", page_icon="🤖", layout="wide")

# FastAPI address
HOST = "http://localhost:8181/answer"

# Load logo
logo_path = "images/logo.png"
logo = Image.open(logo_path)

# Session state initialization
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'messages' not in st.session_state:
    st.session_state.messages = [] 
if 'last_user_input' not in st.session_state:
    st.session_state.last_user_input = None
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []


# Question suggestions by category
SUGGESTED_QUESTIONS = {
    'Cardiology': [
        "What are the symptoms of a heart attack?",
        "How to prevent cardiovascular diseases?",
    ],
    'Dermatology': [
        "What are the signs of melanoma?",
        "How to treat severe acne?",
    ],
    'Neurology': [
        "What are the early signs of Alzheimer's?",
        "How to prevent migraines?",
    ]
}

def get_bot_response(user_input):
    try:
        greeting_terms = ["hello", "hi", "hey", "bonjour", "salut", "coucou", "ça va", "comment vas-tu", 
                         "merci", "thank you", "thanks", "au revoir", "bye"]
        is_greeting = any(term in user_input.lower() for term in greeting_terms)

        documents = []
        doc_contents = []

        if not is_greeting:
            with st.spinner(" Processing..."):
                sources_response = requests.post(
                    "http://localhost:8181/get_sources",
                    json={"question": user_input, "language": "en"},
                    timeout=15
                )
            
            if sources_response.status_code == 200:
                documents = sources_response.json()
                if documents:
                    for i, doc in enumerate(documents):
                        content = doc["page_content"]
                        source = doc["metadata"].get("source", "Unknown")
                        focus_area = doc["metadata"].get("focus_area", "Unknown")
                        answer = doc["metadata"].get("answer", "")
                        score = doc["metadata"].get("similarity_score", None)
                        doc_contents.append({
                            "index": i+1,
                            "content": content,
                            "answer": answer,
                            "source": source,
                            "focus_area": focus_area,
                            "score": score
                        })
                else:
                    st.warning(" No relevant documents found, response generated with general knowledge.")
            else:
                st.error(f" Error retrieving sources: {sources_response.status_code}")

        data = {
            "question": user_input,
            "language": "en",
            "documents": documents
        }
        
        with st.spinner(" Generating response..."):
            response = requests.post("http://localhost:8181/answer", json=data, timeout=15)
            if response.status_code == 200:
                response_data = response.json()
                raw_message = response_data.get('message', 'No response generated.')
                
                # Ne pas restructurer la réponse, sauf si nécessaire
                new_message = raw_message
                
                # Ajouter les documents uniquement si l’API a inclus "Detailed Answer from Medical Database"
                # et si les scores sont pertinents
                min_similarity_threshold = 0.75  # Même seuil que dans api.py
                has_relevant_docs = any(doc["score"] >= min_similarity_threshold for doc in doc_contents if doc["score"] is not None)
                
                if has_relevant_docs and "**Detailed Answer from Medical Database:**" in raw_message:
                    # Si l’API a déjà inclus les détails, pas besoin de les ajouter à nouveau
                    best_score = max([doc["score"] for doc in doc_contents if doc["score"] is not None], default=None)
                    return new_message, best_score
                elif doc_contents and has_relevant_docs:
                    # Cas rare où l’API n’a pas inclus les détails mais les documents sont pertinents
                    new_message += "\n\n**Detailed Answer from Medical Database:**\n\n"
                    unique_answers = {}
                    for doc in doc_contents:
                        if doc["score"] >= min_similarity_threshold:  # Filtrer par seuil
                            answer_text = doc["answer"] if doc["answer"] else doc["content"]
                            answer_key = answer_text[:100].lower()
                            is_duplicate = False
                            for key in unique_answers:
                                if sum(c1 == c2 for c1, c2 in zip(key, answer_key)) / max(len(key), len(answer_key)) > 0.7:
                                    is_duplicate = True
                                    break
                            if not is_duplicate:
                                unique_answers[answer_key] = {
                                    "text": answer_text,
                                    "source": doc["source"],
                                    "focus_area": doc["focus_area"],
                                    "score": doc["score"]
                                }
                    
                    for i, (_, info) in enumerate(unique_answers.items()):
                        new_message += f"**Information {i+1}:**\n"
                        new_message += f"{info['text']}\n"
                        source_info = f"Source: {info['source']} | Focus Area: {info['focus_area']}"
                        if info['score'] is not None:
                            source_info += f" | Similarity Score: {info['score']:.4f}"
                        new_message += f"*{source_info}*\n\n"
                
                best_score = max([doc["score"] for doc in doc_contents if doc["score"] is not None], default=None) if doc_contents else None
                return new_message, best_score
            else:
                return f"Error generating response: {response.status_code}", None
    except requests.exceptions.RequestException as e:
        return f"API connection error: {str(e)}", None

def display_message(sender, message, score=None, timestamp=None):
    import streamlit as st
    from datetime import datetime
    import re
    
    # Utiliser des styles de bulles de conversation
    is_user = sender == "You"
    
    # Définir les styles pour chaque type de message
    if is_user:
        message_style = """
            background-color: #31333f; 
            border-radius: 18px; 
            padding: 10px 15px; 
            margin: 5px 0;
            max-width: 80%;
            float: right;
            clear: both;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            color: white;
        """
        container_style = "display: flex; justify-content: flex-end; margin: 10px 0;"
        text_align = "right"
    else:
        message_style = """
            background-color: #E6F2FA; 
            border-radius: 18px; 
            padding: 10px 15px; 
            margin: 5px 0;
            max-width: 80%;
            float: left;
            clear: both;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        """
        container_style = "display: flex; justify-content: flex-start; margin: 10px 0;"
        text_align = "left"
    
    # Conteneur principal
    st.markdown(f"<div style='{container_style}'>", unsafe_allow_html=True)
    
    # Nettoyer le message
    cleaned_message = re.sub(r'<[^>]*>', '', message)
    cleaned_message = cleaned_message.replace("</div>", "")
    
    # Afficher la bulle de message
    st.markdown(
        f"""
        <div style="{message_style}">
            <div style="color: #1C6EA4; font-weight: bold; margin-bottom: 5px; text-align: {text_align};">{sender}</div>
            <div>{cleaned_message}</div>
            {f'<div style="font-size: 0.8em; color: #888; text-align: {text_align}; margin-top: 5px;">{timestamp.strftime("%H:%M")}</div>' if timestamp else ''}
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Boutons de feedback seulement pour les messages du bot
    if sender == "AIHealthBot":
        message_index = -1
        for i, (msg_sender, msg, msg_score) in enumerate(st.session_state.messages):
            if msg_sender == sender and msg == message:
                message_index = i
                break
        
        unique_id = f"{message_index}_{datetime.now().timestamp()}"
        
        # Ajouter des boutons HTML personnalisés avec CSS et JavaScript
        st.markdown(f"""
        <style>
            .feedback-button {{
                border: none !important;
                background: none !important;
                font-size: 16px !important;
                margin: 0 2px !important;
                padding: 0 !important;
                cursor: pointer;
            }}
            .feedback-button:hover {{
                opacity: 0.7;
            }}
        </style>
        <div style="display: flex; justify-content: flex-start; margin-top: -10px;">  <!-- Modifier ici -->
            <button class="feedback-button" onclick="handleFeedback('like_{unique_id}')">👍</button>
            <button class="feedback-button" onclick="handleFeedback('dislike_{unique_id}')">👎</button>
        </div>
        <script>
            function handleFeedback(buttonId) {{
                // Envoyer une requête à Streamlit pour mettre à jour l'état
                Streamlit.setComponentValue(buttonId);
            }}
        </script>
        """, unsafe_allow_html=True)
        
        # Gérer les interactions avec les boutons
        feedback = st.session_state.get("feedback", None)
        if feedback == f"like_{unique_id}":
            st.session_state.feedback_history.append(("positive", message))
            st.success("Thank you for your feedback!")
        elif feedback == f"dislike_{unique_id}":
            st.session_state.feedback_history.append(("negative", message))
            st.error("Sorry for this response. We'll improve!")

def show_home():
    col1, col2 = st.columns([0.2, 3])
    with col1:
        st.image(logo, width=100)
    with col2:
        st.markdown("<h1 style='margin-top:-5px; margin-left:20px;'>AIHealthBot</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(45deg, #1d313c, #248fbc);
                padding: 40px; 
                border-radius: 20px; 
                text-align: center; 
                color: white;
                margin: 20px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h2 style="color: white;">Your Intelligent Medical Assistant 🩺</h2>
        <p style="font-size: 1.2em;">Get accurate and quick medical answers powered by AI.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3>Explore our areas of expertise 🏥</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 20px; margin: 30px 0;">
        <div style="
            background: linear-gradient(145deg, #ffffff, #f0f0f0);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
            flex: 1;
            text-align: center;
            max-width: 300px;">
            <h4 style="color: #248fbc;">🫀 Cardiology</h4>
            <p>Expertise in cardiovascular diseases and cardiac prevention</p>
        </div>
        <div style="
            background: linear-gradient(145deg, #ffffff, #f0f0f0);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
            flex: 1;
            text-align: center;
            max-width: 300px;">
            <h4 style="color: #248fbc;">🧠 Neurology</h4>
            <p>Specialization in neurological disorders and brain diseases</p>
        </div>
        <div style="
            background: linear-gradient(145deg, #ffffff, #f0f0f0);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
            flex: 1;
            text-align: center;
            max-width: 300px;">
            <h4 style="color: #248fbc;">👨‍⚕️ General Medicine</h4>
            <p>General consultation and personalized medical follow-up</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
        .stButton button {
            background-color: #248fbc;
            padding: 15px 30px;
            color: white;
            border-radius: 10px;
            font-size: 20px;
            border: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #1d313c;
        }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("Start consultation", use_container_width=True):
        st.session_state.current_page = 'chat'
        st.rerun()

def show_chat():
    col1, col2 = st.columns([0.2, 3])
    with col1:
        st.image(logo, width=100)
    with col2:
        st.markdown("<h1 style='margin-top:-5px; margin-left:20px;'>AIHealthBot</h1>", unsafe_allow_html=True)
    
    # Consultation title
    st.markdown("<h3 style='text-align: center;'>💬 Medical consultation</h3>", unsafe_allow_html=True)
    
    # Chat area with original button styles
    st.markdown("""
    <style>
        .stButton button {
            background-color: #248fbc;
            padding: 15px 30px;
            color: white;
            border-radius: 10px;
            font-size: 20px;
            border: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #1d313c;
        }
    </style>
    """, unsafe_allow_html=True)

    chat_container = st.container()
    with chat_container:
        for sender, message, score in st.session_state.messages:
            display_message(sender, message, score, datetime.now())

    # Input area
    st.text_input(
        "Ask your medical question...",
        key="user_input",
        placeholder="Example: What are the symptoms of the flu?",
        on_change=submit_message
    )
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_user_input = None
            st.rerun()
    with col2:
        if st.button("🏠 Back to home", use_container_width=True):
            st.session_state.current_page = 'home'
            st.rerun()

def submit_message():
    if "user_input" in st.session_state and st.session_state.user_input.strip():
        user_message = st.session_state.user_input.strip()

        if user_message != st.session_state.last_user_input:
            st.session_state.last_user_input = user_message
            st.session_state.messages.append(("You", user_message, None))
            bot_response, score = get_bot_response(user_message)
            st.session_state.messages.append(("AIHealthBot", bot_response, score))
            st.session_state.user_input = ""

# Simplified sidebar
with st.sidebar:
    st.markdown("<h3>About</h3>", unsafe_allow_html=True)
    st.write("""
    AIHealthBot uses artificial intelligence to provide reliable medical information.
    
    Note: This assistant does not replace a professional medical consultation.
    """)

if st.session_state.current_page == 'chat':
    show_chat()
else:
    show_home()