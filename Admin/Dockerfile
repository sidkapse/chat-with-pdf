FROM python:3.11
#Paste in AWS credentials to run Admin successfully
ENV AWS_REGION=us-east-1
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN python.exe -m pip install --upgrade pip
COPY . ./


# Health check
HEALTHCHECK CMD curl --fail http://localhost:$PORT || exit 1

ENV PORT=8083
ENV BUCKET_NAME=sidd-poc-bucket-us
EXPOSE $PORT
ENTRYPOINT [ "streamlit", "run", "admin.py", "--server.port=8083", "--server.address=0.0.0.0" ]
RUN echo $AWS_PROFILE
