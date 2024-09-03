import requests

url = "http://localhost:8000/query"

query = """SELECT * FROM titanic LIMIT 5;"""


response = requests.post(url, json={"query": query}, timeout=10)

print(response, response.text)
