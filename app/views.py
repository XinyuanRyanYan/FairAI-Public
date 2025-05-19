from flask import Blueprint, render_template
from flask import Flask, jsonify, request, g
from flask import session
from flask_session import Session
from pytest import console_main
from app import app
from app import APP_STATIC
# from .logisticRegression import Logistic_Regression
from .fairAI import fairAI
from .processConfig import ProcessConfig
from .LogisticRegressionC1 import LRChapter1
from .LRExplainer import lrExplainer
from os import path

@app.route('/')
def index():
    # clear session
    for key in list(session.keys()):
        session.pop(key)

    session['count'] = 0
    session['fairAIObj'] = ''
    session['configs'] = ''
    session['LRChapter1Obj'] = ''
    session['LREObj'] = ''
    session['LRModel'] = ''

    session['fairAIObj'] = fairAI(path.join(APP_STATIC,"uploads/creditScore.csv"))
    # session['fairAIObj'] = fairAI(path.join(APP_STATIC,"uploads/bank_5000.csv"))
    session['configs'] = ProcessConfig()
    return render_template('index.html')

# the following are the old logistic regression model part
# send the train data and test data to the frontend
@app.route('/data', methods=['POST', 'GET'])
def getData():
    train_data = session['LRModel'].train_data
    test_data = session['LRModel'].test_data

    return jsonify({'trainData': train_data, 'testData': test_data})

# train the model and send the coes to the frontend 
@app.route('/train', methods=['POST', 'GET'])
def getModelCoes():
    session['LRModel'].train_model()
    return jsonify({'a': session['LRModel'].coe, 'b': session['LRModel'].intercept})

# train the model and send the coes to the frontend 
@app.route('/test', methods=['POST', 'GET'])
def getPrediction():
    prediction = session['LRModel'].test_model()
    return jsonify({'prediction': prediction})

# {node: {x: _x, y: _y, id: nodeId}, train: train}
@app.route('/addData', methods=['POST', 'GET'])
def addData():
    paras = request.get_json()
    session['LRModel'].add_item(paras['node'], paras['train'])
    return ('', 204) 

# {node: {x: _x, id: nodeId}, train: train}
@app.route('/modifyData', methods=['POST', 'GET'])
def modifyData():
    paras = request.get_json()
    session['LRModel'].modify_item(paras['node'], paras['train'])
    return ('', 204) 


# {node: {id: nodeId}, train: train}
@app.route('/deleteData', methods=['POST', 'GET'])
def deleteData():
    paras = request.get_json()
    session['LRModel'].delete_item(paras['node'], paras['train'])
    return ('', 204)


# generate the train and test data based on the form information
# {'number': 0.5, 'features': ['employment', 'dependents', 'age'], 'sensitiveAttr': 'gender', 'ratio': 0.5}
@app.route('/getTrainTest', methods=['POST', 'GET'])
def getTrainTest():
    paras = request.get_json()
    # new a object
    session['LRChapter1Obj'] = LRChapter1()
    session['LRChapter1Obj'].split(split_ratio=paras['ratio'], keep_features = paras['features'])
    return jsonify(session['LRChapter1Obj'].get_input_data_info())

# return the basic information of the raw data
@app.route('/getRawDataOInfo', methods=['POST'])
def getRawDataOInfo():
    return session['fairAIObj'].get_data_info()

# return the raw data
@app.route('/getTabelData', methods=['POST'])
def getTabelData():
    return session['fairAIObj'].get_table_data()

# return the bar chart data
# {'attribute': 'gender', 'type': 'train'}
@app.route('/attrDtb', methods=['POST'])
def getAttrDtb():
    paras = request.get_json()
    attribute = paras['attribute']
    data_type = paras['type']
    res = session['fairAIObj'].get_attr_dtb_data(attribute, data_type)   # {data: [{v: , l: }], attribute: 'gender', continuous: true/false}
    return jsonify(res)

# when first load the website, init the homepage, return all titles of these chapter and the first page
@app.route('/init', methods=['POST'])
def initInterface():
    session['fairAIObj'] = fairAI(path.join(APP_STATIC,"uploads/creditScore.csv"))
    res = {}
    res['titles'] = session['configs'].chapter_title_lst
    res['firstChapter'] = session['configs'].get_chapter(1)
    return jsonify(res)

# get the json file of this chapter
@app.route('/getChapter', methods=['POST'])
def getChapter():
    paras = request.get_json()
    chapter_id = paras['id']
    if chapter_id == 1:
        session['fairAIObj'] = fairAI(path.join(APP_STATIC,"uploads/creditScore.csv"))
    # if chapter_id == 3:
    #     session['fairAIObj'] = fairAI(path.join(APP_STATIC,"uploads/creditScore.csv"))
    if chapter_id == 5:
        # use the bank data
        print('we are using the bank data')
        session['fairAIObj'] = fairAI(path.join(APP_STATIC,"uploads/bank_5000.csv"), is_bank_data=True)
    return jsonify(session['configs'].get_chapter(chapter_id))

# train the  model and get the train result of this data
@app.route('/trainModel', methods=['POST'])
def trainModel():
    paras = request.get_json()
    res = session['fairAIObj'].train_model(paras['modelName'], paras['trainName'])
    print('trainProcessEnd', res)
    return jsonify({'a': 'b'})

# test the  model and get the test result of this data
@app.route('/testModel', methods=['POST'])
def testModel():
    paras = request.get_json()
    res = session['fairAIObj'].test(paras['dataName'], paras['train'], paras['model'])
    return jsonify(res)

# train the model in the chapter 1
@app.route('/trainModelC1', methods=['POST'])
def trainModelC1():
    paras = request.get_json()
    res = session['LRChapter1Obj'].train_model()
    return jsonify(res)

# test the  model and get the test result of this datain the chapter1
@app.route('/testModelC1', methods=['POST'])
def testModelC1():
    paras = request.get_json()
    res = session['LRChapter1Obj'].test_model()
    return jsonify(res)

# get the metric values of four fairness metrics
# {'type': 'Original'/'Reweighing'/'LFR'/'OptimPreproc'; 'inSet': -1(not in VSSet), otherwise, >-1}
# return: if a single fairness metrics: {'SPD': [{'Original': 07}, {} ....], 'DI': [], 'EOD': [], 'AOD': [], 'CF': [[], []]}
@app.route('/getMetrics', methods=['POST'])
def getMetrics():
    paras = request.get_json()
    print(paras)
    type = paras['type']
    inSet = paras['inSet']
    if type == 'Original' and inSet == -1: # use this one when show single fairness metric component
        session['fairAIObj'] = fairAI(path.join(APP_STATIC,"uploads/creditScore.csv"))
        session['fairAIObj'].customize_data(split_ratio=0.68)
        session['fairAIObj'].train_model('LR', 'Original')
        cf_data = session['fairAIObj'].test('test', 'Original', 'LR')['data']  # evaluate the model with the test data
        metrics = session['fairAIObj'].get_test_fair_metrics(type)
        metrics['CF'] = cf_data
        metrics['accuracy'] = session['fairAIObj'].test_accuracies['Original']
        return metrics
    return jsonify(session['fairAIObj'].get_test_fair_metrics(type))

# get the baseline of accuracy and confusion matrix
@app.route('/getBaseAccCF', methods=['POST'])
def getBaseAccCF():
    session['fairAIObj'].train_model('LR', 'Original')
    cf_data = session['fairAIObj'].test('test', 'Original', 'LR')['data']  # evaluate the model with the test data
    metrics = session['fairAIObj'].get_test_fair_metrics('Original')
    metrics['CF'] = cf_data
    metrics['accuracy'] = session['fairAIObj'].test_accuracies['Original']
    return metrics



# get the accuacy
# {'type': 'Original'/'Reweighing'/'LFR'/'OptimPreproc'/..}
@app.route('/getAccuracy', methods=['POST'])
def getAccuracy():
    paras = request.get_json()
    type = paras['type']
    return jsonify(session['fairAIObj'].get_accuracies(type))
    
# get the metric values of two fairness metrics for the training dataset
@app.route('/getTrainMetrics', methods=['POST'])
def getTrainMetrics():
    return jsonify(session['fairAIObj'].get_train_fair_metrics())

@app.route('/getReweighingWeights', methods=['POST'])
def getReweighingWeights():
    return jsonify(session['fairAIObj'].reweighing())

# get the train data after preprocessing according to type
# {'type': 'Reweighing'/'LFR'/'OptimPreproc'}
@app.route('/getPreprocessData', methods=['GET', 'POST'])
def getPreprocData():
    paras = request.get_json()
    type = paras['type']
    print('getPreprocessData', paras)
    return jsonify(session['fairAIObj'].preProcess(type))

# get the train data after preprocessing according to type
# {'type': 'Reweighing'/'LFR'/'OptimPreproc'}
@app.route('/getPostprocessData', methods=['GET', 'POST'])
def getPostprocessData():
    paras = request.get_json()
    type = paras['type']
    return jsonify(session['fairAIObj'].ROC())

# when open the debias chapter, then new a fairObj
@app.route('/startDebias', methods=['POST'])
def startDebias():
    session['fairAIObj'] = fairAI(path.join(APP_STATIC,"uploads/creditScore.csv"))
    session['fairAIObj'].customize_data(split_ratio=0.68)
    session['fairAIObj'].train_model('LR', 'Original')   # train the Logistic Regression model
    test_res = session['fairAIObj'].test('test', 'Original', 'LR')  # evaluate the model with the test data
    res = session['fairAIObj'].get_input_data_info()
    res['output'] = test_res['data']
    return jsonify(res)

# when open the project chapter, then new a fairObj
@app.route('/startProject', methods=['POST'])
def startProject():
    session['fairAIObj'] = fairAI(path.join(APP_STATIC, 'uploads/bank-additional-full.csv'), True)
    session['fairAIObj'].customize_data(split_ratio=0.7)
    return jsonify(session['fairAIObj'].get_input_data_info())

# verify if the password is true
@app.route('/verifyPwd', methods=['POST'])
def verifyPwd():
    res = {'res': 'no'}
    paras = request.get_json()
    pwd = paras['pwd']
    if pwd.strip()=='mktg6650':
        res['res'] = 'yes'

    return jsonify(res)

###############################################################################################
## the following view function is used handle the Logistic Regression Explain visual components
###############################################################################################

# get the train and test data
@app.route('/getLREData', methods=['POST'])
def getLREData():
    # print('get the train data')
    session['LREObj'] = lrExplainer()
    # print(session['LREObj'])
    res = {'train': session['LREObj'].train_dict_lst, 'test': session['LREObj'].test_dict_lst}

    return jsonify(res)

# train data and the get the coes of LR
@app.route('/trainLREData', methods=['POST'])
def trainLREData():
    # session['count'] += 1
    # print('count:', session['count'])
    paras = request.get_json()
    train_data = paras['data']
    res = session['LREObj'].train_model(train_data)
    # print('start to train the data')
    # print(session['LREObj'])
    
    return jsonify(res)

# test model and the get the prediction
@app.route('/testLREData', methods=['POST'])
def testLREData():
    paras = request.get_json()
    test_data = paras['data']
    res = session['LREObj'].test_model(test_data)
    return jsonify(res)