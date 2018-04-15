import requests
import os
import cv2
import json

addr = 'http://localhost:3001'
post_url = addr+"/upload"
content_type = 'image/jpeg'
headers = {'content-type': content_type}

for filename in os.listdir("people_to_register"):
    files = {'photo': open(os.path.join("people_to_register", filename), 'rb')}
    data = {"name": filename[:-9]}
    response = requests.post(post_url, files=files)
    print(response.status_code)