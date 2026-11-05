
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from rag.retriever import retriever


# 1. Initialize LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# 2. Create prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the provided context.

If the answer is not available in the context,
say that you don't know based on the provided documents.

Context:
{context}

Question:
{question}

Answer:
""")


# 3. Format retrieved documents
def format_docs(documents):
    return "\n\n".join(
        doc.page_content for doc in documents
    )


# 4. Build RAG chain using LCEL
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# 5. Chat loop
if __name__ == "__main__":

    print("RAG Chatbot started! Type 'exit' to quit.")

    while True:
        question = input("\nYou: ")

        if question.lower() in ["exit", "quit"]:
            break

        answer = rag_chain.invoke(question)

        print("\nAssistant:", answer)
