from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import (GoogleGenerativeAI, HarmBlockThreshold, HarmCategory)
from stembeddings import STEmbeddings
from dotenv import load_dotenv
import os


# get chain
def get_qa_chain():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    emb_path = os.getenv("EMB_PATH")
    db_path = os.getenv("DB_PATH")
    prompt_path = os.getenv("PROMPT_PATH")
    model_name = os.getenv("MODEL_NAME")
    document_prompt_path = os.getenv("DOC_PROMPT_PATH")

    # get llm
    llm = GoogleGenerativeAI(model=model_name, google_api_key=api_key, temparture=0.6,
                             safety_settings=
                             {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                              HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                              HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                              HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE})

    # get embeddings
    emb_model = STEmbeddings(model_path=emb_path)

    # get vector database
    vectorstore = Chroma(embedding_function=emb_model, persist_directory=db_path)

    # get prompt
    with open(prompt_path, 'r') as file:
        prompt_template = file.read()

    with open(document_prompt_path, 'r') as file:
        document_prompt_template = file.read()

    retriever = vectorstore.as_retriever(search_kwargs={'k': 10})

    PROMPT = PromptTemplate(template=prompt_template,
                            input_variables=['input', 'context']
                            )
    document_prompt = PromptTemplate(
        input_variables=["page_content", "web_url", 'pub_date'],
        template=document_prompt_template
    )

    combine_documents_chain = create_stuff_documents_chain(
        llm=llm,
        document_variable_name="context",
        prompt=PROMPT,
        document_prompt=document_prompt
    )

    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_documents_chain
    )

    return chain


if __name__ == "__main__":
    pass