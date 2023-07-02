import os
import requests

url = "https://deepfacecloudapiunibatesi.azurewebsites.net//identify"

payload = dict(identity='Beppe Marotta', info='Calciatore Inter')

resp = requests.post(url=url, data=payload, files={
        'img': open(os.path.join('testdataset/', 'single', '26.jpg'), 'rb')})
print(f'- {resp.json()}')

