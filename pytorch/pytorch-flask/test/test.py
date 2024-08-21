import requests

resp = requests.post("http://localhost:5000/predict", files={"file": open('test2.png','rb')})

print(resp.text)