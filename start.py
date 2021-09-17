import asyncio
from aiohttp import web
import uvloop
from pathlib import Path
import simplejson as json
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from classifier_test_api import train_models, predict_models_for_rows
from datetime import datetime

Model = namedtuple('Model', 'model_name date parameters_learning quality_indicators')
models_db_file = 'models.json'
models_db = []

# dir for all data
dir_data = Path('data').resolve()
if not dir_data.exists():
    dir_data.mkdir()

# dir for models_save_data
dir_tmp = Path('data/models_saved_data').resolve()
if not dir_tmp.exists():
    dir_tmp.mkdir()

# dir for datasets
dir_dataset = Path('data/datasets').resolve()
if not dir_dataset.exists():
    dir_dataset.mkdir()

# dir for models
dir_models = Path('data/models').resolve()
if not dir_models.exists():
    dir_models.mkdir()

# dir for models_db (model_name - parameters for model)
dir_dbs = Path('dbs').resolve()
if not dir_dbs.exists():
    dir_dbs.mkdir()

models_db_file = dir_dbs/models_db_file
if models_db_file.exists() and models_db_file.stat().st_size:
    with open(models_db_file) as json_file:
        models_db = json.load(json_file, object_hook=lambda d: namedtuple('Model', d.keys())(*d.values()))


def save_db_to_file(db):
    """ save new state of db (after changes) to file """
    with open(models_db_file, 'w') as outfile:
        json.dump([model._asdict() for model in models_db], outfile)


def conv_time(stamp):
    """ conversion unix timestamp to str format """
    value = datetime.fromtimestamp(stamp)
    return value.strftime('%Y-%m-%d %H:%M:%S')


async def list(request):
    """ out list models and datasets """
    dict_answer = {'models': [item[1]+' '+item[0]+str(item[2:]) for item in models_db],
                   'datasets': [conv_time(d.stat().st_atime)+' '+str(d.name) for d in Path('data/datasets/').glob('*')],
                   }
    return web.json_response(dict_answer)


async def upload(request):
    filename, size = await load_csv_file(request)
    if size > 0:
        return web.json_response({"result": "ok"})
    else:
        return web.json_response({"result": "Not loaded! Saved 0 bytes!"})


async def load_csv_file(request):
    reader = await request.multipart()
    field = await reader.next()
    filename = field.filename
    # You cannot rely on Content-Length if transfer is chunked.
    size = 0
    with open(dir_dataset / filename, 'wb') as f:
        while True:
            chunk = await field.read_chunk()  # 8192 bytes by default.
            if not chunk:
                break
            size += len(chunk)
            f.write(chunk)
    return filename, size


def check_train_parameters(check_params):
    result_check = {}
    full_list_params = ['dataset', 'model_name', 'enum_int', 'algo_name', 'params']
    for parameter in full_list_params:
        if parameter in check_params:
            if parameter == 'params':
                for k in check_params[parameter]:
                    result_check[k] = check_params[parameter][k]
                continue
            result_check[parameter] = check_params[parameter]
        else:
            result_check[parameter] = F'{parameter} is not setting and has value - None!'
    result_check['model_name'] = str(Path(dir_models/result_check['model_name']).resolve())

    # check exists fatal errors in setting
    # fatal is true if not exists dataset or algo_name is not correct setting
    fatal = False
    if Path(dir_dataset / result_check['dataset']).resolve().exists():
        result_check['dataset'] = str(Path(dir_dataset / result_check['dataset']).resolve())
    else:
        result_check['dataset'] = result_check['dataset'] + '- not found! Fatal error!'
        fatal = True
    if result_check['algo_name'] not in ['autokeras']:
        result_check['algo_name'] = result_check['algo_name'] + 'is not correct! Fatal error!'
        fatal = True
    if fatal:
        return result_check, False
    return result_check, True


def save_model_with_params(res):
    if len(models_db):
        for item in models_db:
            if item[0] == res['model_name']:
                models_db.remove(item)
                break
    models_db.append(Model(res['model_name'], res['date'], res['parameters_for_learning'], res['indicators']))
    save_db_to_file('models')


async def train(request):
    global loop
    pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()-1)
    try:
        train_params = await request.json()
    except:
        return web.Response(text='Check your request, please!')
    checked_params, status_check = check_train_parameters(train_params)
    if not status_check:
        return web.json_response({'Error': "fatal error. Check your parameters:", **checked_params})

    # training and save model
    results = await loop.run_in_executor(pool, start_train, checked_params)
    if results['status'] == 'ok':
        save_model_with_params(results)
        return web.json_response({'model_name': results['model_name'], 'status': 'trained', 'date': results['date'], 'indicators': results['indicators']})
    else:
        return web.json_response({'model_name': results['model_name'], 'parameters': results})


def start_train(params):
    results = train_models(params)
    return results


async def predict(request):
    global loop
    pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()-1)
    try:
        predict_params = await request.json()
        rows = predict_params['rows']
        models_list = predict_params['models']
    except Exception as e:
        return web.Response(text='Error! Check your request, please!\n'+str(e.__class__))
    if not len(rows):
        return web.Response(text='Error: not setting rows for testing!')
    if not len(models_list):
        return web.Response(text='Error: not setting models for testing!')
    results = await loop.run_in_executor(pool, predict_rows_on_models, models_list, rows)
    if not results:
        return web.Response(text='Error: In your request for /test - Not found correct rows!')
    return web.json_response(results)


def predict_rows_on_models(model_list, rows):
    results = predict_models_for_rows(model_list, rows)
    return results


asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
loop = asyncio.get_event_loop()
app = web.Application(loop=loop)

app.router.add_route('GET', '/', list)
app.router.add_route('POST', '/upload/', upload)
app.router.add_route('POST', '/train/', train)
app.router.add_route('POST', '/predict/', predict)

if __name__ == '__main__':
    web.run_app(app)


