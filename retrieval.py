from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

def create_conversational_chain(vectorstore: Chroma) -> ConversationalRetrievalChain:
    """
    Создаёт цепочку ConversationalRetrievalChain с пользовательским промптом.
    Ассистент отвечает как высококвалифицированный бизнес-консультант в сфере стоматологических услуг,
    предоставляя подробные ответы и бизнес-инсайты.
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory.output_key = "answer"  

    prompt_template = (
        "Ты высококвалифицированный бизнес-консультант, специализирующийся на стоматологических услугах. "
        "Основываясь исключительно на предоставленном контексте, ответь на следующий вопрос в 2–3 абзацах. "
        "Кроме того, проанализируй историю разговора и предоставь ключевые тенденции и рекомендации.\n\n"
        "Context: {context}\n"
        "Chat History: {chat_history}\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    
    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=prompt_template
    )
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain
