FROM python:3.12-slim

RUN apt-get update && apt-get install -y sqlite3

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["sh","-c", "python3 /app/create_data.py && tail -f /dev/null"]
