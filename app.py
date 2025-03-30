import csv
import os
import sys
import tempfile
import time
import base64
from datetime import datetime
from io import StringIO, BytesIO

import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Imports pour Kokoro TTS
from kokoro import KPipeline
import soundfile as sf

system_prompt = """You are an well-being coach expert in conversation with a human-like direct style. 

Here is the provided context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the conversation history and the user's latest question:

Conversation History:
{conversation_history}

Latest Question:
{question}

Provide a factual answer to this questions using only the above provided context and conversation history. Do not include any external knowledge or references or assumptions not present in the given context. 

Do not give an opinion or remarks. Do not make introductions or conclusions. Do not make explanations or summaries. Do not make assumptions. Do not say that you are answering from a provided context.

Use one to three sentences maximum (it can be way less). Be direct in your answer and do not repeat yourself. Don't use enumerations. Keep the answer short, concice and accurate to the provided context with the highest fidelity.

Make sure your answers are consistent with your previous answers in the conversation history.

If the question is irrelevant to your role as a well-being coach, just answer : "It's not my role to answer this question".
If the question is relevant to your role but don't find the answer in the context or if you have a doubt, don't give explnations, only say :"I can't give give you an accurate answer to this question yet".
If any question is about you or refering to your behavior, just answer that you are a coach and that you try to help.

Answer:"""

# Initialisation de l'historique de conversation dans session_state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

def get_redis_store() -> RedisVectorStore:
    """Gets or creates a Redis vector store for caching embeddings.

    Creates an Ollama embeddings object using the nomic-embed-text model and initializes
    a Redis vector store with cosine similarity metric for storing cached question-answer pairs.

    Returns:
        RedisVectorStore: A Redis vector store configured with Ollama embeddings and
            metadata schema for storing answers.

    Raises:
        RedisConnectionError: If unable to connect to Redis.
    """
    embeddings = OllamaEmbeddings(
        model="bge-m3:567m",
    )
    return RedisVectorStore(
        embeddings,
        config=RedisConfig(
            index_name="cached_contents",
            redis_url="redis://localhost:6379",
            distance_metric="COSINE",
            metadata_schema=[
                {"name": "answer", "type": "text"},
            ],
        ),
    )


def create_cached_contents(uploaded_file: UploadedFile) -> list[Document]:
    """Creates cached question-answer pairs from an uploaded CSV file.

    Takes an uploaded CSV file containing question-answer pairs, converts them to Document
    objects and adds them to a Redis vector store for caching.

    Args:
        uploaded_file: A Streamlit UploadedFile object containing the CSV data with
            'question' and 'answer' columns.

    Returns:
        list[Document]: List of Document objects created from the CSV rows.

    Raises:
        ValueError: If CSV is missing required 'question' or 'answer' columns.
        RedisConnectionError: If unable to add documents to Redis vector store.
    """
    data = uploaded_file.getvalue().decode("utf-8")
    csv_reader = csv.DictReader(StringIO(data))

    docs = []
    for row in csv_reader:
        docs.append(
            Document(page_content=row["question"], metadata={"answer": row["answer"]})
        )
    vector_store = get_redis_store()
    vector_store.add_documents(docs)
    st.success("Cache contents added!")


def query_semantic_cache(query: str, n_results: int = 1, threshold: float = 75.0):
    """Queries the semantic cache for similar questions and returns cached results if found.

    Args:
        query: The search query text to find relevant cached results.
        n_results: Maximum number of results to return. Defaults to 1.
        threshold: Minimum similarity score threshold (0-100) for returning cached results.
            Defaults to 80.0.

    Returns:
        list: List of tuples containing matched Documents and their similarity scores if
            matches above threshold are found. None if no matches above threshold.

    Raises:
        RedisConnectionError: If there are issues connecting to Redis.
    """
    vector_store = get_redis_store()
    results = vector_store.similarity_search_with_score(query, k=n_results)

    if not results:
        return None

    match_percentage = (1 - abs(results[0][1])) * 100
    if match_percentage >= threshold:
        return results
    return None


def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """Processes an uploaded PDF file by converting it to text chunks.

    Takes an uploaded PDF file, saves it temporarily, loads and splits the content
    into text chunks using recursive character splitting.

    Args:
        uploaded_file: A Streamlit UploadedFile object containing the PDF file

    Returns:
        A list of Document objects containing the chunked text from the PDF

    Raises:
        IOError: If there are issues reading/writing the temporary file
    """
    # Store uploaded file as a temp file
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name)  # Delete temp file

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)


def get_vector_collection() -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage.

    Creates an Ollama embedding function using the nomic-embed-text model and initializes
    a persistent ChromaDB client. Returns a collection that can be used to store and
    query document embeddings.

    Returns:
        chromadb.Collection: A ChromaDB collection configured with the Ollama embedding
            function and cosine similarity space.
    """
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="bge-m3:567m",
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(all_splits: list[Document], file_name: str):
    """Adds document splits to a vector collection for semantic search.

    Takes a list of document splits and adds them to a ChromaDB vector collection
    along with their metadata and unique IDs based on the filename.

    Args:
        all_splits: List of Document objects containing text chunks and metadata
        file_name: String identifier used to generate unique IDs for the chunks

    Returns:
        None. Displays a success message via Streamlit when complete.

    Raises:
        ChromaDBError: If there are issues upserting documents to the collection
    """
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Data added to the vector store!")


def query_collection(prompt: str, n_results: int = 20):
    """Queries the vector collection with a given prompt to retrieve relevant documents.

    Args:
        prompt: The search query text to find relevant documents.
        n_results: Maximum number of results to return. Defaults to 10.

    Returns:
        dict: Query results containing documents, distances and metadata from the collection.

    Raises:
        ChromaDBError: If there are issues querying the collection.
    """
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results


def format_conversation_history(history):
    """Formate l'historique de conversation pour l'inclure dans le prompt.
    
    Args:
        history: Liste de tuples (question, réponse)
        
    Returns:
        str: Historique formaté pour le prompt
    """
    if not history:
        return ""
    
    formatted_history = ""
    for i, (question, answer) in enumerate(history):
        formatted_history += f"User: {question}\nAssistant: {answer}\n\n"
    
    return formatted_history


def call_llm(context: str, prompt: str, conversation_history=None, show_thinking: bool = False, timer_placeholder=None):
    """Calls the language model with context, conversation history and prompt to generate a response.

    Uses Ollama to stream responses from a language model by providing context and a
    question prompt. The model uses a system prompt to format and ground its responses appropriately.

    Args:
        context: String containing the relevant context for answering the question
        prompt: String containing the user's question
        conversation_history: List of previous conversation turns
        show_thinking: Boolean indicating whether to show the model's thinking process
        timer_placeholder: Placeholder Streamlit pour afficher le chronomètre

    Yields:
        String chunks of the generated response as they become available from the model

    Raises:
        OllamaError: If there are issues communicating with the Ollama API
    """
    # Marquer le début de la génération
    start_time = time.time()
    
    # Formater l'historique de conversation
    formatted_history = format_conversation_history(conversation_history) if conversation_history else ""
    
    response = ollama.chat(
        model="deepseek-r1:8b-llama-distill-q4_K_M",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt.format(context=context, conversation_history=formatted_history, question=prompt),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    
    # Pour le traitement des balises <think>
    buffer = ""
    in_think_block = False
    
    for chunk in response:
        # Mettre à jour le chronomètre à chaque chunk reçu
        if timer_placeholder:
            elapsed = time.time() - start_time
            timer_placeholder.metric("⏱️", f"{elapsed:.1f}s")
        
        if chunk["done"] is False:
            content = chunk["message"]["content"]
            buffer += content
            
            if not show_thinking:
                # Traiter le contenu pour filtrer les blocs <think>...</think>
                while "<think>" in buffer or "</think>" in buffer:
                    # Si on trouve une balise d'ouverture
                    if "<think>" in buffer and not in_think_block:
                        # Extraire le contenu avant la balise et l'envoyer
                        parts = buffer.split("<think>", 1)
                        if parts[0]:
                            yield parts[0]
                        buffer = parts[1]
                        in_think_block = True
                    
                    # Si on trouve une balise de fermeture
                    elif "</think>" in buffer and in_think_block:
                        # Extraire le contenu après la balise et continuer
                        parts = buffer.split("</think>", 1)
                        buffer = parts[1] if len(parts) > 1 else ""
                        in_think_block = False
                    
                    # Si pas de balise ou traitement incomplet, sortir de la boucle
                    else:
                        break
                
                # Envoyer le contenu accumulé s'il n'est pas dans un bloc think
                if not in_think_block and buffer and "<think>" not in buffer:
                    yield buffer
                    buffer = ""
            else:
                # Si l'affichage du processus de réflexion est activé, envoyer tel quel
                yield content
        else:
            # Traiter tout contenu restant
            if buffer and not in_think_block and not show_thinking:
                yield buffer
            
            # Enregistrer et afficher le temps total final
            total_time = time.time() - start_time
            if timer_placeholder:
                timer_placeholder.metric("⏱️", f"{total_time:.1f}s")
            st.session_state.total_time = total_time
            break


def re_rank_cross_encoders(documents: list[str], show_thinking: bool = False) -> tuple[str, list[int]]:
    """Re-ranks documents using a cross-encoder model for more accurate relevance scoring.

    Uses the MS MARCO MiniLM cross-encoder model to re-rank the input documents based on
    their relevance to the query prompt. Returns the concatenated text of the top 3 most
    relevant documents along with their indices.

    Args:
        documents: List of document strings to be re-ranked.
        show_thinking: Boolean indicating whether to show the reranking process.

    Returns:
        tuple: A tuple containing:
            - relevant_text (str): Concatenated text from the top 3 ranked documents
            - relevant_text_ids (list[int]): List of indices for the top ranked documents

    Raises:
        ValueError: If documents list is empty
        RuntimeError: If cross-encoder model fails to load or rank documents
    """
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    ranks = encoder_model.rank(prompt, documents, top_k=10)
    
    # Afficher les détails du reranking uniquement si show_thinking est activé
    if show_thinking:
        st.write(ranks)
    
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])
    
    # Afficher le texte pertinent uniquement si show_thinking est activé
    if show_thinking:
        st.write(relevant_text)
        st.divider()
        
    return relevant_text, relevant_text_ids


def text_to_speech(text, voice="af_heart"):
    """
    Convertit du texte en audio avec Kokoro TTS.
    
    Args:
        text: Le texte à convertir en audio
        voice: La voix à utiliser pour la synthèse
    
    Returns:
        bytes: Les données audio en format wav
    """
    if not text.strip():
        return None
    
    # Nettoyer le texte avant de le passer à Kokoro
    # En supprimant les balises <think> si présentes
    text = text.replace("<think>", "").replace("</think>", "")
    
    # Initialiser le pipeline Kokoro
    pipeline = KPipeline(lang_code='a')
    
    # Créer un BytesIO pour stocker les données audio
    audio_data = BytesIO()
    
    # Générer l'audio
    try:
        generator = pipeline(text, voice=voice)
        for i, (gs, ps, audio) in enumerate(generator):
            # Prendre seulement le premier segment audio généré
            if i == 0:
                # Écrire les données audio dans le buffer
                sf.write(audio_data, audio, 24000, format='WAV')
                break
        
        # Réinitialiser le pointeur au début du buffer
        audio_data.seek(0)
        return audio_data.read()
    except Exception as e:
        st.error(f"Erreur lors de la génération audio: {str(e)}")
        return None

def create_audio_player_html(audio_bytes):
    """
    Crée un élément audio HTML à partir des données audio.
    
    Args:
        audio_bytes: Les données audio en bytes
    
    Returns:
        str: Le HTML de l'élément audio
    """
    if audio_bytes is None:
        return ""
        
    # Encoder les données audio en base64
    b64 = base64.b64encode(audio_bytes).decode()
    
    # Créer un élément audio HTML
    html = f"""
    <audio controls autoplay style="width: 100%;">
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        Votre navigateur ne supporte pas l'élément audio.
    </audio>
    """
    return html


if __name__ == "__main__":
    # Initialiser les variables de session pour le chronomètre
    if 'total_time' not in st.session_state:
        st.session_state.total_time = 0
    
    # Initialiser l'historique de conversation s'il n'existe pas
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Initialiser les variables pour conserver les données entre les recharges
    if 'last_thinking_data' not in st.session_state:
        st.session_state.last_thinking_data = None
    if 'last_relevant_ids' not in st.session_state:
        st.session_state.last_relevant_ids = None
    if 'last_relevant_text' not in st.session_state:
        st.session_state.last_relevant_text = None
    if 'last_results' not in st.session_state:
        st.session_state.last_results = None
    if 'last_audio_data' not in st.session_state:
        st.session_state.last_audio_data = None
    if 'submit_clicked' not in st.session_state:
        st.session_state.submit_clicked = False
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0
    
    # Document Upload Area
    with st.sidebar:
        st.set_page_config(page_title="Coach Conversationnel RAG")
        uploaded_file = st.file_uploader(
            "**📑 Upload PDF files for QnA**",
            type=["pdf", "csv"],
            accept_multiple_files=False,
            help="Upload csv for cached results only",
        )
        upload_option = st.radio(
            "Upload options:",
            options=["Primary", "Cache"],
            help="Choose Primary for uploading document for QnA.\n\nChoose Cache for uploading cached results",
        )
        
        if (
            uploaded_file
            and upload_option == "Primary"
            and uploaded_file.name.split(".")[-1] == "csv"
        ):
            st.error("CSV is only allowed for 'Cache' option.")
            sys.exit(1)

        process = st.button(
            "⚡️ Process",
        )
        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )

            if upload_option == "Cache":
                all_splits = create_cached_contents(uploaded_file)
            else:
                all_splits = process_document(uploaded_file)
                add_to_vector_collection(all_splits, normalize_uploaded_file_name)
        
        # Bouton pour réinitialiser la conversation
        if st.button("🔄 Nouvelle conversation"):
            st.session_state.conversation_history = []
            st.session_state.last_thinking_data = None
            st.session_state.last_relevant_ids = None
            st.session_state.last_relevant_text = None
            st.session_state.last_results = None
            st.session_state.last_audio_data = None
            st.session_state.submit_clicked = False
            # Incrémenter pour forcer le rechargement du widget
            st.session_state.input_key += 1
            st.rerun()

    # Question and Answer Area
    st.header("🗣️ Conversation avec votre Coach")
    
    # Afficher l'historique de conversation
    conversation_container = st.container()
    with conversation_container:
        for i, (question, answer) in enumerate(st.session_state.conversation_history):
            st.markdown(f"**Vous**: {question}")
            st.markdown(f"**Coach**: {answer}")
            st.divider()
    
    # Fonction pour gérer la soumission et vider l'entrée après traitement
    def handle_submit():
        st.session_state.submit_clicked = True
        # Stocker temporairement le contenu de l'entrée utilisateur
        if "temp_input" not in st.session_state:
            st.session_state.temp_input = st.session_state.get(f"question_input_{st.session_state.input_key}", "")
        # Incrémenter la clé pour forcer la recréation du widget avec une valeur vide
        st.session_state.input_key += 1
    
    # Zone de texte pour la nouvelle question avec une clé dynamique
    prompt_key = f"question_input_{st.session_state.input_key}"
    prompt = st.text_area("**Posez une question à votre coach:**", key=prompt_key)
    
    # Nouvelle interface avec bouton plus petit et zone pour le chronomètre et TTS
    cols = st.columns([2, 1, 1, 1])
    with cols[0]:
        ask = st.button("🔥 Envoyer", use_container_width=False, on_click=handle_submit)
    with cols[1]:
        show_thinking = st.checkbox("💭 Afficher la réflexion", value=False, 
                                  help="Montre ou cache le processus de réflexion du modèle (balises <think>)", 
                                  key="show_thinking_checkbox")
    with cols[2]:
        enable_tts = st.checkbox("🔊 Audio TTS", value=False,
                               help="Activer la lecture audio de la réponse",
                               key="enable_tts_checkbox")
    with cols[3]:
        # Afficher le chronomètre s'il y a un temps total
        if st.session_state.total_time > 0:
            st.metric("⏱️", f"{st.session_state.total_time:.1f}s")
        else:
            st.empty()
    
    # Afficher l'audio de la dernière réponse s'il existe
    if st.session_state.last_audio_data is not None and enable_tts:
        st.markdown(create_audio_player_html(st.session_state.last_audio_data), unsafe_allow_html=True)
    
    # Déplacement des expanders après cette section pour les afficher après le traitement
    
    # Utiliser la valeur temporairement stockée si le bouton a été cliqué
    user_input = ""
    if st.session_state.submit_clicked and "temp_input" in st.session_state:
        user_input = st.session_state.temp_input
        # Réinitialiser pour la prochaine utilisation
        del st.session_state.temp_input
        st.session_state.submit_clicked = False
    
    if user_input:
        # Réinitialiser le temps total
        st.session_state.total_time = 0
        
        # Container pour le chronomètre en temps réel
        timer_container = st.empty()
        timer_container.markdown("⏱️ **0.0s**")
        
        start_time = time.time() # Démarrer le chronomètre global ici
        
        # Fonction pour mettre à jour le chronomètre
        def update_timer():
            elapsed = time.time() - start_time
            timer_container.markdown(f"⏱️ **{elapsed:.1f}s**")
        
        # Première mise à jour
        update_timer()
        
        # Afficher la question de l'utilisateur immédiatement
        st.markdown(f"**Vous**: {user_input}")
        
        cached_results = query_semantic_cache(query=user_input)

        if cached_results:
            # Afficher les résultats du cache
            response_text = cached_results[0][0].metadata["answer"].replace("\\n", "\n")
            st.markdown(f"**Coach**: {response_text}")
            
            # Ajouter à l'historique de conversation
            st.session_state.conversation_history.append((user_input, response_text))
            
            # Générer et afficher l'audio si activé
            if enable_tts:
                audio_bytes = text_to_speech(response_text)
                if audio_bytes:
                    st.markdown(create_audio_player_html(audio_bytes), unsafe_allow_html=True)
                    st.session_state.last_audio_data = audio_bytes
            
            # Mise à jour finale du chrono pour les résultats du cache
            total_time = time.time() - start_time
            timer_container.markdown(f"⏱️ **{total_time:.1f}s**")
            st.session_state.total_time = total_time
        else:
            # Mettre à jour le chronomètre avant chaque étape
            update_timer()
            results = query_collection(prompt=user_input)
            # Sauvegarder pour l'affichage ultérieur
            st.session_state.last_results = results
            
            # Mettre à jour à nouveau
            update_timer()
            
            context = results.get("documents")[0]
            
            if not context:
                st.write("No results found.")
                sys.exit(1)

            # Mettre à jour à nouveau
            update_timer()
            
            # Collecter les données de réflexion si show_thinking est activé
            thinking_data = []
            
            # Passer le paramètre show_thinking à la fonction re_rank_cross_encoders
            relevant_text, relevant_text_ids = re_rank_cross_encoders(context, show_thinking=False)
            
            # Sauvegarder pour l'affichage ultérieur
            st.session_state.last_relevant_text = relevant_text
            st.session_state.last_relevant_ids = relevant_text_ids
            
            # Mettre à jour à nouveau
            update_timer()
            
            # Préparation de la réponse du coach
            st.markdown("**Coach**: ")
            
            # Placeholder pour la réponse
            response_placeholder = st.empty()
            
            # Container pour l'audio
            audio_placeholder = st.empty()
            
            # Générer et afficher la réponse
            response_text = ""
            
            # Pour collecter le thinking process
            thinking_process = ""
            
            # Utiliser directement ollama.chat pour plus de contrôle
            response = ollama.chat(
                model="deepseek-r1:8b-llama-distill-q4_K_M",
                stream=True,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt.format(
                            context=relevant_text, 
                            conversation_history=format_conversation_history(st.session_state.conversation_history),
                            question=user_input
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_input,
                    },
                ],
            )
            
            # Pour le traitement des balises <think>
            buffer = ""
            in_think_block = False
            
            # Pour stocker le texte filtré pour TTS (sans balises think)
            filtered_text_for_tts = ""
            
            for chunk in response:
                # Mettre à jour le chrono à chaque chunk
                update_timer()
                
                if chunk["done"] is False:
                    content = chunk["message"]["content"]
                    buffer += content
                    
                    if not show_thinking:
                        # Traiter le contenu pour filtrer les blocs <think>...</think>
                        while "<think>" in buffer or "</think>" in buffer:
                            # Si on trouve une balise d'ouverture
                            if "<think>" in buffer and not in_think_block:
                                # Extraire le contenu avant la balise et l'envoyer
                                parts = buffer.split("<think>", 1)
                                if parts[0]:
                                    response_text += parts[0]
                                    filtered_text_for_tts += parts[0]  # Ajouter au texte filtré
                                    response_placeholder.markdown(response_text)
                                buffer = parts[1]
                                in_think_block = True
                                # Collecter pour le thinking process
                                thinking_process += "<think>"
                            
                            # Si on trouve une balise de fermeture
                            elif "</think>" in buffer and in_think_block:
                                # Extraire le contenu après la balise et continuer
                                parts = buffer.split("</think>", 1)
                                # Collecter pour le thinking process
                                thinking_process += parts[0] + "</think>"
                                buffer = parts[1] if len(parts) > 1 else ""
                                in_think_block = False
                            
                            # Si pas de balise ou traitement incomplet, sortir de la boucle
                            else:
                                break
                        
                        # Envoyer le contenu accumulé s'il n'est pas dans un bloc think
                        if not in_think_block and buffer and "<think>" not in buffer:
                            response_text += buffer
                            filtered_text_for_tts += buffer  # Ajouter au texte filtré
                            response_placeholder.markdown(response_text)
                            buffer = ""
                    else:
                        # Si l'affichage du processus de réflexion est activé
                        # Ajouter au processus de réflexion complet
                        thinking_process += content
                        
                        # Traiter le contenu pour filtrer les balises <think> même si show_thinking est activé
                        # pour l'affichage dans la réponse principale
                        if "<think>" in content or "</think>" in content or in_think_block:
                            # Si le contenu contient <think> ou est dans un bloc think
                            if "<think>" in content:
                                parts = content.split("<think>", 1)
                                if parts[0]:
                                    response_text += parts[0]
                                    filtered_text_for_tts += parts[0]
                                in_think_block = True
                                if len(parts) > 1:
                                    # Traiter le reste après <think>
                                    remaining = parts[1]
                                    if "</think>" in remaining:
                                        think_parts = remaining.split("</think>", 1)
                                        in_think_block = False
                                        if len(think_parts) > 1 and think_parts[1]:
                                            response_text += think_parts[1]
                                            filtered_text_for_tts += think_parts[1]
                            elif "</think>" in content and in_think_block:
                                parts = content.split("</think>", 1)
                                in_think_block = False
                                if len(parts) > 1 and parts[1]:
                                    response_text += parts[1]
                                    filtered_text_for_tts += parts[1]
                        else:
                            # Si le contenu ne contient pas de balises think
                            response_text += content
                            filtered_text_for_tts += content
                            
                        response_placeholder.markdown(response_text)
                else:
                    # Traiter tout contenu restant
                    if buffer and not in_think_block and not show_thinking:
                        response_text += buffer
                        filtered_text_for_tts += buffer
                        response_placeholder.markdown(response_text)
                    
                    # Sauvegarder le thinking process pour affichage ultérieur
                    st.session_state.last_thinking_data = thinking_process
                    
                    # Ajouter à l'historique de conversation
                    st.session_state.conversation_history.append((user_input, filtered_text_for_tts))
                    
                    # Générer et afficher l'audio si activé
                    if enable_tts and filtered_text_for_tts:
                        audio_bytes = text_to_speech(filtered_text_for_tts)
                        if audio_bytes:
                            audio_placeholder.markdown(create_audio_player_html(audio_bytes), unsafe_allow_html=True)
                            # Sauvegarder l'audio pour affichage ultérieur
                            st.session_state.last_audio_data = audio_bytes
                    
                    # Afficher les expanders immédiatement après avoir terminé la génération
                    if show_thinking:
                        if st.session_state.last_results is not None:
                            with st.expander("Voir les documents récupérés", expanded=False):
                                st.write(st.session_state.last_results)
                        
                        if st.session_state.last_relevant_ids is not None and st.session_state.last_relevant_text is not None:
                            with st.expander("Voir les documents les plus pertinents", expanded=False):
                                st.write(st.session_state.last_relevant_ids)
                                st.write(st.session_state.last_relevant_text)
                        
                        if st.session_state.last_thinking_data is not None:
                            with st.expander("Voir le processus de réflexion du modèle", expanded=False):
                                st.write(st.session_state.last_thinking_data)
                    
                    # Enregistrer et afficher le temps total final
                    break
            
            # Mise à jour finale du chronomètre
            total_time = time.time() - start_time
            timer_container.markdown(f"⏱️ **{total_time:.1f}s**")
            st.session_state.total_time = total_time
    else:
        # Afficher les expanders même si aucune nouvelle question n'est posée
        if show_thinking:
            if st.session_state.last_results is not None:
                with st.expander("Voir les documents récupérés", expanded=False):
                    st.write(st.session_state.last_results)
            
            if st.session_state.last_relevant_ids is not None and st.session_state.last_relevant_text is not None:
                with st.expander("Voir les documents les plus pertinents", expanded=False):
                    st.write(st.session_state.last_relevant_ids)
                    st.write(st.session_state.last_relevant_text)
            
            if st.session_state.last_thinking_data is not None:
                with st.expander("Voir le processus de réflexion du modèle", expanded=False):
                    st.write(st.session_state.last_thinking_data)
