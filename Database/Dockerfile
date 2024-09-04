FROM python:3.12-slim

RUN useradd -m nonrootuser && \
    apt-get update && \
    apt-get install -y sqlite3 && \
    rm -rf /var/lib/apt/lists/*

USER nonrootuser

WORKDIR /home/nonrootuser

COPY . /home/nonrootuser/app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["sh","-c", "python3 /app/create_data.py && tail -f /dev/null"]
