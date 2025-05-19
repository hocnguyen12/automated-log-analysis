FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

ENV STREAMLIT_VERSION=1.44.1 
ENV PANDAS_VERSION=2.2.3
ENV NUMPY_VERSION=1.25.2
ENV SKLEARN_VERSION=1.3.2 
ENV SENTENCE_TRANSFORMERS_VERSION=4.0.1 
ENV FAISS_VERSION=1.8.0 
ENV JOBLIB_VERSION=1.3.2 
ENV MATPLOTLIB_VERSION=3.8.2
ENV NESTED_LAYOUT_VERSION=0.1.4

RUN pip install --no-cache-dir \
    streamlit==${STREAMLIT_VERSION} \
    pandas==${PANDAS_VERSION} \
    numpy==${NUMPY_VERSION} \
    scikit-learn==${SKLEARN_VERSION} \
    sentence-transformers==${SENTENCE_TRANSFORMERS_VERSION} \
    faiss-cpu==${FAISS_VERSION} \
    joblib==${JOBLIB_VERSION} \
    matplotlib==${MATPLOTLIB_VERSION} \
    streamlit-nested-layout==${NESTED_LAYOUT_VERSION}

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
