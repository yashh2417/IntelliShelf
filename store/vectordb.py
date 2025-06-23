# Now we can load the persisted database from disk, and use it as normal.
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
)

persist_directory = '../chromadb'

vectorstore = Chroma(persist_directory=persist_directory,
                  embedding_function=gemini_embeddings)

retriever = vectorstore.as_retriever()
# print(retriever)

docs = retriever.get_relevant_documents("what is return policy of intellishelf?")

system_prompt = (
    "You are an assistant for question answering tasks. "
    "Use the following pieces of retrieved context to answer the question "
    "If you don't know the answer, say that you don't know."
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)



chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

model = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", convert_system_message_to_human=True)



question_answering_chain = create_stuff_documents_chain(model, chat_prompt)

from langchain.chains import create_retrieval_chain

rag_chain = create_retrieval_chain(retriever, question_answering_chain)

answer = rag_chain.invoke({"input": "what is return policy of intellishelf?"})["answer"]

print(answer)