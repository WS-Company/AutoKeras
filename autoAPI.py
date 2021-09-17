""" реализация интерфейса командной строки  """

import fire
import simplejson as json
import requests
from pathlib import Path

BASE_URL = 'http://0.0.0.0:8080/'


def list():
    source = requests.get(BASE_URL)
    return source.json()


def upload(name_dataset):
    url = f'{BASE_URL}upload/'
    files = {Path(name_dataset).name: open(name_dataset, 'rb')}
    source = requests.post(url, files=files)
    return source.json()


def train(model_name, dataset, algo_name, enum_int, params):
    url = f'{BASE_URL}train/'
    print(model_name, dataset, algo_name, enum_int, params)
    data = {'model_name': model_name,
            'dataset': dataset,
            'algo_name': algo_name,
            'enum_int': enum_int,
            'params': params
            }
    data_json = json.dumps(data)
    source = requests.post(url, data=data_json)
    return source.text


def predict(rows, models):
    url = f'{BASE_URL}predict/'
    data = {'rows': rows, 'models': models}
    data_json = json.dumps(data)
    print(data_json)
    source = requests.post(url, data=data_json)
    return source.text


if __name__ == '__main__':
    fire.Fire()
