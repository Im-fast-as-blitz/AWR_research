import argparse

from models import SAC

import json

def get_model_config(model_name):
    try:
        with open(f'../configs/{model_name}.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print("Файл не найден!")
    except json.JSONDecodeError:
        print("Ошибка декодирования JSON!")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


def get_model(model_name, global_params):
    model_params = get_model_config(model_name) | global_params
    if model_name == "SAC":
        return SAC(**model_params)
    else:
        raise "Неизвестный алгоритм :("


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True)
    parser.add_argument('-m', '--model', type=str, required=True)


    return parser.parse_args()