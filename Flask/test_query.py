import requests
import json

url = "http://localhost:8000/query"

query = """SELECT * FROM titanic LIMIT 5;"""


response = requests.post(url, json={"query": query})

print(response, response.text)
