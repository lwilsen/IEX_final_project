FROM python:3.12-slim 

RUN useradd -m nonrootuser

USER nonrootuser

WORKDIR /home/nonrootuser

COPY . /home/nonrootuser/app

RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get update && apt-get install ffmpeg libsm6 libxext6  -y && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 8001

CMD ["streamlit", "run", "Home.py", "--server.port", "8001", "--server.address", "0.0.0.0"]
