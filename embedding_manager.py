import os
import hashlib
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader

# Импортируем nltk и загружаем токенизатор предложений.
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

def load_documents_from_folder(folder_path: str) -> List[Document]:
    """
    Загружает документы из указанной папки.
    Поддерживаемые форматы: PDF, Word (.docx) и текст (.txt).
    """
    documents = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file.lower().endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file.lower().endswith('.txt'):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            continue  
        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = file
        documents.extend(docs)
    return documents

def get_file_hash(file_path: str) -> str:
    """
    Вычисляет MD5-хэш файла для отслеживания изменений.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def sentence_level_chunk_document(doc: Document, min_tokens: int = 256, max_tokens: int = 512) -> List[Document]:
    """
    Разбивает один документ на чанки по границам предложений.
    Каждый чанк имеет приблизительно от min_tokens до max_tokens слов.
    На чанки поделил по Sentence-chunking. После куча тестов, именно он дает лучший результат в эмбединге.
    """
    sentences = sent_tokenize(doc.page_content)
    chunks = []
    current_chunk_sentences = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        if current_word_count + sentence_word_count > max_tokens and current_word_count >= min_tokens:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentence]
            current_word_count = sentence_word_count
        else:
            current_chunk_sentences.append(sentence)
            current_word_count += sentence_word_count
    
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
    
    new_docs = []
    for idx, chunk in enumerate(chunks):
        new_metadata = dict(doc.metadata)
        new_metadata["chunk_index"] = idx
        new_doc = Document(page_content=chunk, metadata=new_metadata)
        new_docs.append(new_doc)
    return new_docs

def sentence_level_chunk_documents(documents: List[Document], min_tokens: int = 256, max_tokens: int = 512) -> List[Document]:
    """
    Применяет разбиение на чанки для списка документов.
    Если документ содержит меньше min_tokens слов, он остаётся без изменений.
    """
    new_documents = []
    for doc in documents:
        word_count = len(doc.page_content.split())
        if word_count < min_tokens:
            new_documents.append(doc)
        else:
            chunks = sentence_level_chunk_document(doc, min_tokens, max_tokens)
            new_documents.extend(chunks)
    return new_documents

def update_vector_store(folder_path: str, persist_directory: str = "chroma_db") -> Chroma:
    """
    Создаёт или обновляет векторное пространство (Chroma) из документов в указанной папке.
    """
    documents = load_documents_from_folder(folder_path)
    documents = sentence_level_chunk_documents(documents, min_tokens=256, max_tokens=512)
    
    embeddings = OpenAIEmbeddings()  # Использует OPENAI_API_KEY из окружения

    if os.path.exists(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        vectorstore.add_documents(documents)
    else:
        vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    vectorstore.persist()
    return vectorstore
