import os
from math import ceil
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
# get_chat_history_text
def generate_dashboards_and_commentary():
    sales_path = os.path.join("Datasets", "sales_data.csv")
    sentiment_path = os.path.join("Datasets", "customer_sentiment.csv")
    
    try:
        df_sales = pd.read_csv(sales_path, encoding="utf-8")
    except Exception as e:
        return None, f"Ошибка при чтении sales_data.csv: {e}", ""
    
    try:
        df_sentiment = pd.read_csv(sentiment_path, encoding="utf-8")
    except Exception as e:
        return None, f"Ошибка при чтении customer_sentiment.csv: {e}", ""
    
    figs = []

    # Линейный график продаж по датам
    if "Date" in df_sales.columns and "Sales" in df_sales.columns:
        df_sales["Date"] = pd.to_datetime(df_sales["Date"], errors="coerce")
        df_sales_sorted = df_sales.sort_values("Date")
        fig_line = px.line(df_sales_sorted, x="Date", y="Sales", title="Тренд продаж по датам")
        figs.append(fig_line)

    # Столбчатая диаграмма продаж по регионам
    if "Region" in df_sales.columns and "Sales" in df_sales.columns:
        fig_bar = px.bar(df_sales, x="Region", y="Sales", title="Продажи по регионам", color="Region")
        figs.append(fig_bar)

    # Круговая диаграмма распределения продаж по продуктам
    if "Product" in df_sales.columns and "Sales" in df_sales.columns:
        product_group = df_sales.groupby("Product")["Sales"].sum().reset_index()
        fig_pie = px.pie(product_group, names="Product", values="Sales", title="Распределение продаж по продуктам")
        figs.append(fig_pie)

    # Столбчатая диаграмма количества отзывов по тональности
    if "Sentiment" in df_sentiment.columns:
        sentiment_count = df_sentiment["Sentiment"].value_counts().reset_index()
        sentiment_count.columns = ["Sentiment", "Count"]
        fig_sent = px.bar(sentiment_count, x="Sentiment", y="Count", title="Количество отзывов по тональности", color="Sentiment")
        figs.append(fig_sent)

    # Гистограмма частоты отзывов по датам
    if "Date" in df_sentiment.columns:
        df_sentiment["Date"] = pd.to_datetime(df_sentiment["Date"], errors="coerce")
        df_sentiment_sorted = df_sentiment.sort_values("Date")
        fig_sent_timeline = px.histogram(df_sentiment_sorted, x="Date", title="Частота отзывов по датам")
        figs.append(fig_sent_timeline)
    
    # Определяем число графиков и создаём комбинированный dashboard с правильными specs.
    n = len(figs)
    if n == 0:
        combined_fig = None
    else:
        rows = ceil(n / 2)
        specs = []
        for i in range(rows):
            row_specs = []
            for j in range(2):
                idx = i * 2 + j
                if idx < n:
                    trace_type = figs[idx].data[0].type if figs[idx].data else "xy"
                    if trace_type == "pie":
                        row_specs.append({"type": "domain"})
                    else:
                        row_specs.append({"type": "xy"})
                else:
                    row_specs.append({})
            specs.append(row_specs)
        
        combined_fig = make_subplots(rows=rows, cols=2, specs=specs,
                                     subplot_titles=[f.layout.title.text for f in figs])
        for i, fig in enumerate(figs):
            row = (i // 2) + 1
            col = (i % 2) + 1
            for trace in fig.data:
                combined_fig.add_trace(trace, row=row, col=col)
            # Обновляем подписи осей, если заданы
            if hasattr(fig.layout, "xaxis") and fig.layout.xaxis.title.text:
                combined_fig.update_xaxes(title_text=fig.layout.xaxis.title.text, row=row, col=col)
            if hasattr(fig.layout, "yaxis") and fig.layout.yaxis.title.text:
                combined_fig.update_yaxes(title_text=fig.layout.yaxis.title.text, row=row, col=col)
        combined_fig.update_layout(height=400 * rows, showlegend=False, title_text="Дашборд по данным CSV")
    
    total_sales = df_sales["Sales"].sum() if "Sales" in df_sales.columns else 0
    avg_sales = df_sales["Sales"].mean() if "Sales" in df_sales.columns else 0
    sales_summary = f"sales_data.csv: {len(df_sales)} строк, сумма продаж = {total_sales}, средняя продажа = {avg_sales:.2f}."
    
    total_reviews = len(df_sentiment)
    pos = (df_sentiment["Sentiment"]=="Positive").sum() if "Sentiment" in df_sentiment.columns else 0
    neu = (df_sentiment["Sentiment"]=="Neutral").sum() if "Sentiment" in df_sentiment.columns else 0
    neg = (df_sentiment["Sentiment"]=="Negative").sum() if "Sentiment" in df_sentiment.columns else 0
    sentiment_summary = f"customer_sentiment.csv: {total_reviews} отзывов, Положительных: {pos}, Нейтральных: {neu}, Отрицательных: {neg}."
    
    summary_text = sales_summary + "\n" + sentiment_summary

    # Генерируем подробный комментарий через LLM
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
    commentary = llm.predict(prompt)
    
    return combined_fig, commentary, summary_text
