import boto3
import streamlit as st
import os
import uuid

##S3 Client
s3_client = boto3.client("s3", region_name='us-east-1')
BUCKET_NAME = os.getenv("BUCKET_NAME")


##Bedrock
from langchain_aws import BedrockEmbeddings

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name='us-east-1')

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

##Split text in chuncks
##Text Splitter
##What is RecursiveCharacterTextSplitter
## Example
## Line 1: Everything is going good in ECV and my life.
## Line 2: ECV is a good company and give good experience.
##Chunk 1: Everything is going good in ECV and my life.
##Chunk 2: ECV and my life. ECV is a good company and give good experience.
##As we noticed Chunk 2 is having some words from Line 1 combined with line 2 so that context can be maintained.

from langchain.text_splitter import RecursiveCharacterTextSplitter

#Pdf loader
from langchain_community.document_loaders import PyPDFLoader

## Import FAISS
from langchain_community.vectorstores import FAISS

## Split the pages / text into chunks

def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def create_vector_store(request_id, documents):
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    file_name=f"{request_id}.bin"
    folder_path="/tmp/"
    vectorstore_faiss.save_local(index_name=file_name, folder_path = folder_path)

    ##upload to s3
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")

    return True


def main():
    st.write("This is Admin Site for Chat with PDF.")
    st.write("Developed and Designed by Sidd.")
    uploaded_file = st.file_uploader("Choose a file", "pdf")
    if uploaded_file is not None:
        request_id = uuid.uuid4()
        st.write(f"Request Id: {request_id}")
        saved_file_name = f"{request_id}.pdf"
        ##Open the file
        with open(saved_file_name, mode="wb") as w:
            w.write(uploaded_file.getvalue())

        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()

        st.write(f"Total Pages: {len(pages)}")

        ##Split Text
        splitted_docs = split_text(pages, 1000, 200)
        # st.write(f"Splitted Docs length: {len(splitted_docs)}")
        # st.write("==============")
        # st.write(splitted_docs[0])
        # st.write("==============")
        # st.write(splitted_docs[1])
        # st.write("==============")
        # st.write(splitted_docs[2])

        st.write("Creating the vector store")
        result = create_vector_store(request_id, splitted_docs)

        if result:
            st.write("PDF vectorized successfully and stored in Vector DB")
        else:
            st.write("Error in PDF Vectorizing, please check logs.")


if __name__=="__main__":
    main()


##docker build -t admin-pdf-reader-v2 .
##docker run -e BUCKET_NAME=sidd-poc-bucket -v 'C:\Users\Administrator\.aws\:/root/.aws/' -p 8083:8083 -it admin-pdf-reader
##docker run -e BUCKET_NAME=sidd-poc-bucket -it -p 8083:8083 --volume C:\Users\Administrator\.aws:/root/.aws/ admin-pdf-reader-v2
##docker run -e BUCKET_NAME=sidd-poc-bucket --mount type=bind,source="c:\Users\Administrator\.aws\",target="root/.aws/" -p 8083:8083 -it admin-pdf-reader

#docker run -e BUCKET_NAME=sidd-poc-bucket -it -p 8083:8083 --volume C:\Users\temp:/root/temp admin-pdf-reader

##docker run -it -p 8083:8083 admin-pdf-reader-v2


#docker exec -it fc1fe5885fa2 CMD.exe