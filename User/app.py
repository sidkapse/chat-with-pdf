import boto3
import streamlit as st
import os
import uuid

##S3 Client
s3_client = boto3.client("s3", region_name='us-east-1')
BUCKET_NAME = os.getenv("BUCKET_NAME")


##Bedrock
from langchain_aws import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock

##
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name='us-east-1')

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

folderPath = f"${os.getcwd()}"

## Import FAISS
from langchain_community.vectorstores import FAISS

def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename="my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename="my_faiss.pkl")

def get_llm():
    llm = Bedrock(model_id="amazon.titan-text-express-v1", client=bedrock_client, model_kwargs={"maxTokenCount": 512})
    return llm

def get_response(llm, vector_store, question):
    #create Prompt/template
    prompt_template = """Use the following pieces of information to answer the user's question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Context: {context} Question: {question} Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": question})
    print(answer['result'])
    return answer['result']
    

def main():
    st.header("This is User Site for Chat with PDF using Bedrock, RAG")
    load_index()
    # dir_list = os.listdir(os.getcwd())
    # st.write(f"Files and Directories in {os.getcwd()}")
    # st.write(dir_list)

    ##Create a store/index
    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path= os.getcwd(),
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    #st.write("Index created successfully")
    question = st.text_input("Ask a question:")
    if st.button("Ask Question"):
        with st.spinner("Querying..."):
            llm = get_llm()

            st.write(get_response(llm, faiss_index, question))
            st.success("Done")

if __name__=="__main__":
    main()


##docker build -t admin-pdf-reader-v2 .
##docker run -e BUCKET_NAME=sidd-poc-bucket -v 'C:\Users\Administrator\.aws\:/root/.aws/' -p 8083:8083 -it admin-pdf-reader
##docker run -e BUCKET_NAME=sidd-poc-bucket -it -p 8083:8083 --volume C:\Users\Administrator\.aws:/root/.aws/ admin-pdf-reader-v2
##docker run -e BUCKET_NAME=sidd-poc-bucket --mount type=bind,source="c:\Users\Administrator\.aws\",target="root/.aws/" -p 8083:8083 -it admin-pdf-reader

#docker run -e BUCKET_NAME=sidd-poc-bucket -it -p 8083:8083 --volume C:\Users\temp:/root/temp admin-pdf-reader

##docker run -it -p 8083:8083 admin-pdf-reader-v2


#docker exec -it fc1fe5885fa2 CMD.exe