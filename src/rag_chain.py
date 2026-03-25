from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


def create_qa_chain(llm, vectorstore):
    prompt_template = """Use the information from the document to answer the question at the end. 
If you don't know the answer, just say that you don't know, definitely do not try to make up an answer.

{context}

Question: {question}
"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=False,
    )
    return qa_chain


def create_conversational_chain(llm, vectorstore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        get_chat_history=lambda h: h,
        return_source_documents=False,
    )
    return conv_chain
