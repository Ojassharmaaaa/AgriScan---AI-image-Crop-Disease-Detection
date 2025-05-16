import requests

url = 'http://127.0.0.1:5000/predict'
file = {'file': open('test\AppleScab3.JPG', 'rb')}

response = requests.post(url, files=file)
print(response.json())
