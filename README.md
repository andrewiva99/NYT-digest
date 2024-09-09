# NYT Digest

[Docker Hub](https://hub.docker.com/repository/docker/andreybg/nyt-digest/general)


NYT Digest is a web application for searching and summarizing articles from The New York Times, powered by **Streamlit**. 
- The application is built on the RAG architecture using **Langchain** capabilities.
- The generative model used is **Gemini 1.5 Flash**, while **Chroma** serves as the vector database for storing and retrieving data.
- The dataset ```data_20240101_20240831.json``` comprises 14,000 records collected via the [NYT API](https://developer.nytimes.com/apis) between January 1, 2024, and August 31, 2024.


You can also run the application using a **Docker container**:

```docker run -p 8501:8501 andreybg/nyt-digest```
