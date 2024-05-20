import asyncio
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl


DB_FAISS_PATH = 'vectorstores/db_faiss'

custom_prompt_template = """
You're tasked with providing a helpful response based on the given context and question.
Accuracy is paramount, so if you're uncertain, it's best to acknowledge that rather than providing potentially incorrect information.

Context: {context}
Question: {question}


Please craft a clear and informative response that directly addresses the question.
Aim for accuracy and relevance, keeping the user's needs in mind.
Response:
"""
# setting up a pipeline for question-answering (QA) retrieval
def set_custom_prompt():
    # Prompt template for QA retrieval for each vector stores
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

"""
This method initializes a question-answering (QA) retrieval chain, which consists of a conditional
    transformer model (CTransformers) for generating responses, a prompt template for constructing
    prompts based on user queries, and a vector store (FAISS) for retrieving relevant documents
    to support the QA process.

    - llm (CTransformers): The conditional transformer model (CTransformers) plays a central role
    in generating responses to user queries. It leverages pre-trained language
    models to understand user questions and produce informative answers.

    - prompt (PromptTemplate): The prompt template (PromptTemplate) defines the structure of prompts
    provided to the QA model. It includes placeholders for the context
    and question, guiding the model on how to interpret and respond to
    user queries effectively.

    - db (FAISS): The vector store (FAISS) serves as a repository of documents or passages that
    the QA model can search through to find relevant information. It enables efficient
    retrieval of documents based on the similarity of their embeddings to the user query.

    The method initializes a RetrievalQA object, which orchestrates the interaction between the
    conditional transformer model, prompt template, and vector store to facilitate QA retrieval.
    The RetrievalQA object is configured with the specified components, including the model,
    prompt template, retriever (vector store), and additional settings for controlling the QA process.

    The initialized RetrievalQA object is returned, ready to be used for responding to user queries
    by retrieving relevant information from the vector store and generating informative answers
    using the conditional transformer model.
"""
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

def load_llm():
    llm = CTransformers(
        model = 'models/llama-2-7b-chat.ggmlv3.q4_0.bin',
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# Model
async def setup_qa_bot():
    hugging_face_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-V2', model_kwargs={'device': 'cpu'})
    faiss_db = FAISS.load_local(DB_FAISS_PATH, embeddings=hugging_face_embeddings)
    llama_model = load_llm()
    qa_prompt = set_custom_prompt()
    qa_retrieval_chain = retrieval_qa_chain(llama_model, qa_prompt, faiss_db)
    return qa_retrieval_chain

# Output
async def final_response(query):
    qa_result = await setup_qa_bot()
    response = await qa_result({'query': query})
    return response

# chainlit 
@cl.on_chat_start
async def start():
    chain = await setup_qa_bot()
    init_message = cl.Message(content="Initializing the system.")
    await init_message.send()
    
    init_message.content = 'Selamat Pagi! Kita Bisa Luar Biasa!. Yuk tanyakkan pertanyaanmu ke UT Buddy.'
    await init_message.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    # if sources:
    #     answer += f"\nSources:" + str(sources)
    # else:
    #     answer += "\nNo sources found"

    await cl.Message(content=answer).send()

if __name__ == "__main__":
    asyncio.run(cl.main())
