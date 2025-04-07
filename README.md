# Dental RAG

Dental RAG — система Retrieval-Augmented Generation (RAG) для поддержки бизнес-консультирования в сфере стоматологических услуг. Приложение использует LangChain, OpenAI, Chroma и Gradio для загрузки, анализа и визуализации документов и данных.

## Особенности проекта

- **Чат на основе документов**: Используется цепочка `ConversationalRetrievalChain` для ответов на вопросы с учетом контекста.
- **Аналитика**: Генерация дашбордов и подробного анализа данных из CSV-файлов.

## Требования

Установите следующие пакеты (см. [requirements.txt](requirements.txt)):
- gradio
- langchain
- langchain-community
- chromadb
- openai
- python-dotenv
- plotly
- pandas
- nltk

## Настройка проекта

1. **Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/YourUsername/Dental-RAG.git
   cd Dental-RAG
