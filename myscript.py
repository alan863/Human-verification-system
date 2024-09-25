import requests

# URL of the website where you want to send the POST request
url = 'http://127.0.0.1:8000/test/'

# Data to be sent in the POST request
data = {
    'key': 'success',
    'username': 'test'
}

 

# Sending the POST request
requests.post(url, json=data )



