from dotenv import load_dotenv
load_dotenv()

import os
import shutil
import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.io as pio
from math import ceil
from embedding_manager import update_vector_store
from retrieval import create_conversational_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from data_analyzer import generate_dashboards_and_commentary
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader


# Определяем глобальные пути.
FILES_FOLDER = "files"
CHROMA_DB = "chroma_db"

os.makedirs(FILES_FOLDER, exist_ok=True)

# Глобальная переменная для хранения чанков, использованных при последнем запросе.
LAST_USED_CHUNKS = []

vectorstore = update_vector_store(FILES_FOLDER, persist_directory=CHROMA_DB)
qa_chain = create_conversational_chain(vectorstore)

def answer_query(query, chat_history):
    """
    Обрабатывает запрос пользователя с помощью цепочки ConversationalRetrievalChain.
    Добавляет ответ в историю чата и сохраняет документы, использованные при генерации.
    """
    global LAST_USED_CHUNKS
    if not query.strip():
        return "", chat_history
    result = qa_chain({"question": query, "chat_history": chat_history})
    answer = result["answer"]
    LAST_USED_CHUNKS = result.get("source_documents", [])
    chat_history.append((query, answer))
    return "", chat_history

def clear_history():
    """
    Очищает историю чата.
    """
    qa_chain.memory.clear()
    return []

def view_used_chunks():
    """
    Для отладки: возвращает только те чанки, которые были использованы при генерации ответа.
    Каждый текст чанка разделяется пустой строкой.
    """
    if not LAST_USED_CHUNKS:
        return "Пока не было истории чата."
    lines = []
    for doc in LAST_USED_CHUNKS:
        content = doc.page_content.strip() if hasattr(doc, "page_content") else str(doc)
        lines.append(content)
    return "\n\n".join(lines)

def upload_file(file_obj):
    """
    Обрабатывает загрузку файла:
      - Сохраняет файл в папке files.
      - Обновляет векторное пространство и цепочку QA.
    """
    if file_obj is None:
        return "Файл не загружен."
    
    file_name = os.path.basename(file_obj.name) if hasattr(file_obj, "name") else str(file_obj)
    dest_path = os.path.join(FILES_FOLDER, file_name)
    
    if hasattr(file_obj, "read"):
        with open(dest_path, "wb") as f:
            f.write(file_obj.read())
    else:
        if os.path.abspath(file_obj) == os.path.abspath(dest_path):
            return f"Файл '{file_name}' уже существует."
        shutil.copy(file_obj, dest_path)
    
    global vectorstore, qa_chain
    vectorstore = update_vector_store(FILES_FOLDER, persist_directory=CHROMA_DB)
    qa_chain = create_conversational_chain(vectorstore)
    
    return f"Файл '{file_name}' загружен и добавлен в векторное пространство."

def get_chat_history_text(chat_history):
    """
    Преобразует историю чата (список кортежей) в форматированную строку.
    """
    if not chat_history:
        return "Нет истории чата."
    lines = []
    for turn in chat_history:
        user, assistant = turn
        lines.append(f"Пользователь: {user}\nАссистент: {assistant}")
    return "\n\n".join(lines)

def analyze_conversation(chat_history):
    """
    Анализирует историю чата и возвращает структурированный и краткий анализ.
    Если история пуста, возвращает соответствующее сообщение.
    """
    conversation_text = get_chat_history_text(chat_history)
    if conversation_text.strip() == "Нет истории чата.":
        return "Нет данных для анализа."
    
    analysis_prompt = (
        "Ты стратегический бизнес-аналитик в сфере стоматологических услуг. Проанализируй приведённый ниже разговор и дай краткий, структурированный ответ в формате:\n\n"
        "1. Основные вопросы и темы (до 3 пунктов).\n"
        "2. Проблемные моменты (до 2 пунктов).\n"
        "3. Рекомендации (до 3 пунктов).\n\n"
        "Ответ должен быть представлен в виде нумерованного или маркированного списка, без избыточных описаний. Не включай дополнительные разделы или повторения.\n\n"
        "Разговор:\n" + conversation_text + "\n\n"
        "Анализ:"
    )
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
    analysis_result = llm.invoke(analysis_prompt)
    return analysis_result


def custom_prompt_response(user_prompt):
    """
    Обрабатывает пользовательский промпт, введённый вручную.
    """
    if not user_prompt.strip():
        return "Пожалуйста, введите промпт."
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
    response = llm.invoke(user_prompt)
    return response

def show_dashboards():
    """
    Генерирует объединённый дашборд, комментарий и текстовое резюме.
    Если файл analysis_summary.txt уже существует, он не перезаписывается.
    После этого векторное пространство обновляется.
    """
    combined_fig, commentary, summary_text = generate_dashboards_and_commentary()
    summary_path = os.path.join(FILES_FOLDER, "analysis_summary.txt")
    
    # Если файл не существует, сохраняем новое резюме.
    if not os.path.exists(summary_path):
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
    
    global vectorstore, qa_chain
    vectorstore = update_vector_store(FILES_FOLDER, persist_directory=CHROMA_DB)
    qa_chain = create_conversational_chain(vectorstore)
    return combined_fig, commentary


def generate_individual_dashboards():
    """
    Собирает отдельные графики по данным CSV и возвращает список графиков, комментарий и текстовое резюме.
    Логика формирования графиков аналогична generate_dashboards_and_commentary, но без объединения через make_subplots.
    """
    sales_path = os.path.join("Datasets", "sales_data.csv")
    sentiment_path = os.path.join("Datasets", "customer_sentiment.csv")
    
    try:
        df_sales = pd.read_csv(sales_path, encoding="utf-8")
    except Exception as e:
        return [], f"Ошибка при чтении sales_data.csv: {e}", ""
    
    try:
        df_sentiment = pd.read_csv(sentiment_path, encoding="utf-8")
    except Exception as e:
        return [], f"Ошибка при чтении customer_sentiment.csv: {e}", ""
    
    figs = []
    
    if "Date" in df_sales.columns and "Sales" in df_sales.columns:
        df_sales["Date"] = pd.to_datetime(df_sales["Date"], errors="coerce")
        df_sales_sorted = df_sales.sort_values("Date")
        fig_line = px.line(df_sales_sorted, x="Date", y="Sales", title="Тренд продаж по датам")
        figs.append(fig_line)
    
    if "Region" in df_sales.columns and "Sales" in df_sales.columns:
        fig_bar = px.bar(df_sales, x="Region", y="Sales", title="Продажи по регионам", color="Region")
        figs.append(fig_bar)
    
    if "Product" in df_sales.columns and "Sales" in df_sales.columns:
        product_group = df_sales.groupby("Product")["Sales"].sum().reset_index()
        fig_pie = px.pie(product_group, names="Product", values="Sales", title="Распределение продаж по продуктам")
        figs.append(fig_pie)
    
    if "Sentiment" in df_sentiment.columns:
        sentiment_count = df_sentiment["Sentiment"].value_counts().reset_index()
        sentiment_count.columns = ["Sentiment", "Count"]
        fig_sent = px.bar(sentiment_count, x="Sentiment", y="Count", title="Количество отзывов по тональности", color="Sentiment")
        figs.append(fig_sent)
    
    if "Date" in df_sentiment.columns:
        df_sentiment["Date"] = pd.to_datetime(df_sentiment["Date"], errors="coerce")
        df_sentiment_sorted = df_sentiment.sort_values("Date")
        fig_sent_timeline = px.histogram(df_sentiment_sorted, x="Date", title="Частота отзывов по датам")
        figs.append(fig_sent_timeline)
    
    total_sales = df_sales["Sales"].sum() if "Sales" in df_sales.columns else 0
    avg_sales = df_sales["Sales"].mean() if "Sales" in df_sales.columns else 0
    sales_summary = f"sales_data.csv: {len(df_sales)} строк, сумма продаж = {total_sales}, средняя продажа = {avg_sales:.2f}."
    
    total_reviews = len(df_sentiment)
    pos = (df_sentiment["Sentiment"]=="Positive").sum() if "Sentiment" in df_sentiment.columns else 0
    neu = (df_sentiment["Sentiment"]=="Neutral").sum() if "Sentiment" in df_sentiment.columns else 0
    neg = (df_sentiment["Sentiment"]=="Negative").sum() if "Sentiment" in df_sentiment.columns else 0
    sentiment_summary = f"customer_sentiment.csv: {total_reviews} отзывов, Положительных: {pos}, Нейтральных: {neu}, Отрицательных: {neg}."
    
    summary_text = sales_summary + "\n" + sentiment_summary

    prompt = (
        "Ты опытный аналитик в сфере стоматологических услуг. Ниже приведены ключевые данные по продажам и отзывам клиентов:\n\n"
        f"{summary_text}\n\n"
        "1. Рассчитай процентное соотношение положительных, нейтральных и отрицательных отзывов.\n"
        "2. Выдели основные негативные комментарии и проблемы.\n"
        "3. Определи сильные стороны компании, опираясь на положительные отзывы.\n"
        "4. Дай подробные рекомендации для повышения удовлетворённости клиентов.\n\n"
        "Предоставь детальный анализ с конкретными цифрами и предложениями."
    )
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
    commentary = llm.invoke(prompt)
    
    return figs, commentary, summary_text

def show_individual_dashboards():
    """
    Вызывает функцию generate_individual_dashboards(), преобразует список графиков в HTML и возвращает его вместе с комментарием.
    """
    figs, commentary, _ = generate_individual_dashboards()
    html_parts = []
    for fig in figs:
        html_parts.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
    html_combined = "<br><hr><br>".join(html_parts)
    return html_combined, commentary

with gr.Blocks(title="RAG Система") as demo:
    with gr.Tab("Чат"):
        gr.Markdown("# RAG (Чат на основе документов)")
        chatbot = gr.Chatbot(label="Чат")
        with gr.Row():
            query_input = gr.Textbox(
                placeholder="Задайте вопрос...",
                show_label=False,
                container=True,
                scale=9
            )
            send_button = gr.Button("⬆", variant="primary", size="sm", scale=1)
            clear_button = gr.Button("🗑️", size="sm", scale=1)
        query_input.submit(fn=answer_query, inputs=[query_input, chatbot], outputs=[query_input, chatbot])
        send_button.click(fn=answer_query, inputs=[query_input, chatbot], outputs=[query_input, chatbot])
        clear_button.click(fn=clear_history, outputs=chatbot)
    
    with gr.Tab("Выгрузка"):
        gr.Markdown("# Выгрузить документы")
        file_input = gr.File(label="Документы добавляются к существующим эмбеддингам", file_types=[".pdf", ".docx", ".txt"])
        upload_button = gr.Button("Загрузить")
        upload_output = gr.Textbox(label="Статус", lines=2)
        upload_button.click(fn=upload_file, inputs=file_input, outputs=upload_output)
    
    with gr.Tab("DBLookup"):
        gr.Markdown("# Чанки, использованные при генерации")
        view_chunks_button = gr.Button("Вывести чанки")
        chunks_output = gr.Textbox(label="Чанки (разделены пустой строкой)", lines=10)
        view_chunks_button.click(fn=view_used_chunks, outputs=chunks_output)
    
    with gr.Tab("Аналитика"):
        gr.Markdown("# Анализ чата")
        analyze_button = gr.Button("Запустить анализ")
        analysis_output = gr.Textbox(label="По истории переписок (Вообще, пока очень косячно. Плюс, пока что не совсем понятный use-case, ибо все запускается локально, в будущем доработаем.)", lines=10)
        analyze_button.click(fn=analyze_conversation, inputs=chatbot, outputs=analysis_output)
    
    with gr.Tab("Дашборды"):
        gr.Markdown("## Дашборды по данным CSV")
        dashboards_plot = gr.Plot(label="Графики")  # Объединённый dashboard
        dashboards_comment = gr.Textbox(label="Комментарий от ИИ", lines=10)
        dashboards_button = gr.Button("Обновить дашборды")
        dashboards_button.click(fn=show_dashboards, inputs=None, outputs=[dashboards_plot, dashboards_comment])
    
demo.launch(share=True)
