import requests

URL = "http://localhost:8000/query"

QUERY = """SELECT * FROM titanic LIMIT 5;"""


response = requests.post(URL, json={"query": QUERY}, timeout=10)

print(response, response.text)
