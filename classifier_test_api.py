import traceback
import pandas
import simplejson as json
from datetime import datetime
from pathlib import Path
#import pickle
import dill
import tensorflow as tf
import autokeras as ak
from sklearn.model_selection import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('ERROR')

def load_model_from_file(name_file):
    """ native method load model """
    loaded_model = tf.keras.models.load_model(name_file, custom_objects=ak.CUSTOM_OBJECTS)
    return loaded_model


def save_model_to_file(model, name_model):
    """ native method save model """
    exp_model = model.export_model()
    try:
        exp_model.save(Path(str(name_model) + ".tf", save_format="tf"))
        print(str(Path(str(name_model)+".tf"))+' - model saved!')
    except:
        exp_model.save(Path(str(name_model) + ".h5"))
        print(str(Path(str(name_model)+".h5"))+' - model saved!')


def load_pickle_model_from_file(name_file):
    """ using pickle - alternative implementation for native save and load models """
    with open(name_file, 'rb') as file:
        loaded_model = dill.load(file)
    return loaded_model


def save_pickle_model_to_file(model, name_model):
    """ using pickle - alternative implementation for native save and load models """
    with open(Path(str(name_model)+".pickle"), 'wb') as file:
        dill.dump(model, file)
    print(str(Path(str(name_model)+".pickle"))+' - model saved!')


def get_sets_from_enum_dataset(name_dataset, split_value):
    """ read dataset and split for train classifier only """
    x_data = pandas.read_csv(name_dataset, header=None)
    y_data = x_data.pop(len(x_data.columns) - 1)
    split_size = int(split_value * len(x_data))
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=split_size, random_state=None, shuffle=False) #random_state=42
    return (x_train, y_train), (x_test, y_test)


def get_sets_from_int_dataset(name_dataset, split_value):
    """ read dataset and prepare data for train regression model only """
    split_size = 1 - split_value
    raw_dataset = pandas.read_csv(name_dataset, header=None)
    target_column = str(len(raw_dataset.columns) - 1)
    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    dataset.columns = [str(i) for i in range(len(dataset.columns))]
    column_names = list(dataset.columns)
    column_names.remove(target_column)
    data_cols = column_names
    data_type = len(data_cols) * ['numerical']
    data_type = dict(zip(data_cols, data_type))
    train_dataset = dataset.sample(frac=split_size, random_state=42)
    test_dataset = dataset.drop(train_dataset.index)
    train_dataset.describe()
    x_train = train_dataset.drop(columns=[target_column])
    y_train = train_dataset[target_column]
    x_test = test_dataset.drop(columns=[target_column])
    y_test = test_dataset[target_column]
    return (x_train, y_train), (x_test, y_test), (data_cols, data_type)


def train_models(params):
    """ train classifier and regression models """

    if 'split' not in params or params['split'] is None: # check and valid params !!!
        params['split'] = 0.2

    model_name = Path(params['model_name']).name

    if params['enum_int'].lower() == 'enum':
        # classifier
        ext_type = '.classifier'

        if 'batch' not in params or params['batch'] is None: # check and valid params !!!
                params['batch'] = 32

        try:
            (x_train, y_train), (x_test, y_test) = get_sets_from_enum_dataset(params['dataset'], params['split'])
            # train model
            model = ak.StructuredDataClassifier(max_trials=params['max_trials'],
                                                project_name=model_name+ext_type,
                                                directory='data/models_saved_data/',
                                                distribution_strategy=tf.distribute.MirroredStrategy()
                                                )
            model.fit(x_train, y_train,
                      epochs=params['epochs'],
                      batch_size=params['batch'],
                      )
        except:
            return "Error_train:" + traceback.format_exc() + ' ' + json.dumps(params)

    elif params['enum_int'].lower() == 'int':
        # regression
        ext_type = '.regress'

        if 'batch' not in params: # check and valid params !!!
                params['batch'] = None

        try:
            (x_train, y_train), (x_test, y_test), (data_cols, data_type) = get_sets_from_int_dataset(params['dataset'],
                                                                                                     params['split'])
            # train model
            model = ak.StructuredDataRegressor(max_trials=params['max_trials'],
                                               column_names=data_cols,
                                               column_types=data_type,
                                               project_name=model_name+ext_type,
                                               directory='data/models_saved_data/',
                                               distribution_strategy=tf.distribute.MirroredStrategy(),
                                               #loss=tf.keras.losses.Huber()
                                               )
            model.fit(x=x_train, y=y_train, epochs=params['epochs'], batch_size=params['batch'])
        except:
            return "Error_train:" + traceback.format_exc() + ' ' + json.dumps(params)

    if 'batch' not in params or params['batch'] is None: # check and valid params !!!
        params['batch'] = 32

    # evaluate model
    try:
        report = 'Accuracy: {accuracy}'.format(accuracy=model.evaluate(x=x_test, y=y_test, batch_size=params['batch']))
    except:
        return "Error_evalute:" + traceback.format_exc() + ' ' + json.dumps(params)

    # save model
    params['model_name'] = params['model_name'] + ext_type
    #save_pickle_model_to_file(model, params['model_name'])
    save_model_to_file(model, params['model_name']) # this way yet working unstable

    # prepare  report
    params['dataset'] = Path(params['dataset']).name
    params['model_name'] = Path(params['model_name']).name
    result_js = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'ok',
        'model_name': Path(params['model_name']).name,
        'parameters_for_learning': json.dumps(params),
        'indicators': str(report)
    }
    return result_js


def predict_models_for_rows(model_list, rows):
    """ select model and prepare params for run test """
    if not len(rows):    # for any case
        return False
    results_for_models_predicts = []
    for mdl in model_list:
        results_for_models_predicts.append(find_models_and_predict(mdl, rows))
    results_for_models_predicts.append({'rows': rows})
    return results_for_models_predicts


def find_models_and_predict(model_name, rows):
    """ seek all saved models in different formats: .pickle, .h5 , .tf """
    results_predicts = {}
    list_models = [str(d) for d in Path('data/models/').glob(model_name+'*')]
    if not len(list_models):
        results_predicts[Path(model_name).name] = 'Error! Model Not Found!'
        return results_predicts
    for mdl_name in list_models:
        results_predicts[Path(model_name).name] = predict_model_for_rows(mdl_name, rows)
    return results_predicts


def predict_model_for_rows(mdl_name, rows_for_predict):
    """ load model and getting predict for rows  """
    result = {}
    if type(rows_for_predict) is dict:
        x_test = pandas.DataFrame([rows_for_predict[r] for r in rows_for_predict])
    elif type(rows_for_predict) is list:
        x_test = pandas.DataFrame([r for r in rows_for_predict])
    try:
        if Path(mdl_name).suffix == '.pickle':
            model = load_pickle_model_from_file(mdl_name)
        else:
            model = load_model_from_file(mdl_name) # tf and h5 models - yet not working stable

        try:
            result = model.predict(x_test)
            print(result)
            return [str(r[0]) for r in result]
        except Exception as e:
            result['error_predict'] = ' '.join([str(e.__class__), str(rows_for_predict)])
    except Exception as er:
        result['error_model_load'] = ' '.join([str(er.__class__), mdl_name])
    return result


