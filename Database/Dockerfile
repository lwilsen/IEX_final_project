FROM python:3.12-slim

# install sqlite
RUN apt-get update && \
    apt-get install -y sqlite3 && \
    rm -rf /var/lib/apt/lists/*

#set working directory
WORKDIR /app

#copy directory files
COPY . /app

# install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#command to run create_data.py and keep the container running (so data can be added)
CMD ["sh","-c", "python3 /app/create_data.py && tail -f /dev/null"]
