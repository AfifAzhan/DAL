# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:47:27 2022

@author: RAN
"""

import requests
headers = {'Content-Type': 'application/json', 'Accept':'application/json'}
# URL
url = 'http://localhost:5000/api'

json={'exp': [622, 0.717, 150, 0.8, 26, -0.8, 2.7, 3.5, 76]}

r = requests.post(url,json={'exp': [622, 0.717, 150, 0.8, 26, -0.8, 2.7, 3.5, 76]})
# Change the value of experience that you want to test
#json={"exp": [622, 0.717, 150, 0.8, 26, -0.8, 2.7, 3.5, 76]}

print(r.json())



