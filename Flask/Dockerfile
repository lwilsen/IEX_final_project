FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader averaged_perceptron_tagger

EXPOSE 5001

CMD ["python", "-m", "flask", "--app", "app.py", "run", "--host=0.0.0.0"]
