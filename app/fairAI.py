from cgi import test
from cmath import inf
from lib2to3.pgen2.pgen import DFAState
from operator import le
from pyexpat import model
from random import seed
from sqlite3 import Row
from statistics import mode
import sys
from telnetlib import AO
from app import APP_STATIC
from os import path

from sklearn import utils
from sklearn.feature_selection import SelectFdr
from soupsieve import select
from urllib3 import Retry
sys.path.insert(1, "../")  

import numpy as np
np.random.seed(0)
import pandas as pd

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import LFR
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
# from optim_preproc_helpers.adversarialDebiasing import AdversarialDebiasing
from .optim_preproc_helpers.optim_preproc import OptimPreproc
from .optim_preproc_helpers.opt_tools import OptTools

# from aif360.algorithms.inprocessing import PrejudiceRemover

from .optim_preproc_helpers.prejudice_remover import PrejudiceRemover


from aif360.datasets import BankDataset
from aif360.algorithms.postprocessing import RejectOptionClassification

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .utils import check_has_key, find_key_from_lst, load_preproc_data, get_distortion
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from .optim_preproc_helpers.data_preproc_functions import load_preproc_data_bank
from .optim_preproc_helpers.distortion_functions import get_distortion_bank



class fairAI:
    """
    types for preprocess techiques:
        'Reweighing' / 'LFR' / 'OptimPreproc'

    """
    def __init__(self, data_path, is_bank_data = False):
        """load the dataset
        
        Arg:
            data_path (str): the path of the source data
        """
        self.is_bank_data = is_bank_data
        self.df = pd.read_csv(data_path, sep=';') if self.is_bank_data else pd.read_csv(data_path, sep=',')    # raw dataset

        self.dataset_orig = ''          # (BinaryLabelDataset) original dataset
        self.dataset_orig_train = ''    # train part of original dataset
        self.dataset_orig_test = ''     # test part of original dataset
        self.dataset_orig_pred = ''     # original prediction
        self.split_ratio = ''

        self.privileged_groups = ''     # [{'gender': 1}]
        self.unprivileged_groups = ''

        self.metric_orig_train = ''
        self.metric_orig_test = ''
        # self.train_fair_metrics = {'SPD': [], 'DI': []}    # 'SPD': [{'original': 0.95}, {'mitigate': 0.2}...]
        self.miti_train_lst = {}        # {'Reweighing':  train, ... }     value is the data
        self.metric_miti_train_lst = {}     # {'Reweighing': , .. } value is the metric object
        self.test_fair_metrics = {'SPD': {}, 'DI': {}, 'EOD': {}, 'AOD': {}}    # 'SPD': {'Original': 0.95, 'Reweighing': 0.2, ...}
        self.test_accuracies = {}   #{'Reweighing': 0.89, 'Original': 0.6}
        self.model = ''     # the trained model
        self.aif360Model = False    # the trained model is fair 360 or not
        # current model_name and the train_name
        self.model_name = ''
        self.train_name = ''
        # the fair metrics and accuracy for the bank data set
        self.bank_fair_metrics = []
        self.bank_fair_names = []   # train_name + ';' + model_name
        self.bank_fair_accuracies = []
        

    def customize_data(self, split_ratio,  protected_attribute_names=['gender'], keep_features=[]):
        """set the protected attribute, define the privileged/unprivileged groups,
        and split the data into train data and test data (initialize dataset_orig_train dataset_orig_test dataset_orig)

        Args: 
            split_ratio (num: 0-1): the percentage of the train data
            protected_attribute_names (list): protected attributes, by default, gender

        returns:

        """
        self.split_ratio = split_ratio
        if self.is_bank_data:
            protected_attribute_maps = [{1.0: 'married', 0.0: 'unmarried'}]
            self.dataset_orig = BankDataset(
                protected_attribute_names=['marital'],          
                privileged_classes=[['married']], 
                features_to_drop=['campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'],
                categorical_features=['job', 'education', 'default',
                        'housing', 'loan', 'contact', 'month', 'day_of_week'],
                metadata={'protected_attribute_maps': protected_attribute_maps}
            )
            self.privileged_groups = [{'marital': 1}]
            self.unprivileged_groups = [{'marital': 0}]
        else:
            df_copy = copy.deepcopy(self.df)
            if len(keep_features)!=0:
                if 'gender' not in keep_features:
                    keep_features.append('gender')
                print('currently, the preserved features are:', keep_features)
                for feature in df_copy.columns.values.tolist():
                    if feature!='response' and feature not in keep_features:
                       df_copy.drop(feature, axis = 1, inplace=True)

            self.dataset_orig = BinaryLabelDataset(
                df=df_copy,
                label_names=['response'],        
                favorable_label=1,          # response = 1 positive; 
                unfavorable_label=0,        # response = 0 negative;
                protected_attribute_names=protected_attribute_names  
            )
            self.privileged_groups = [{'gender': 1}]
            self.unprivileged_groups = [{'gender': 0}]

        self.dataset_orig_train, self.dataset_orig_test = self.dataset_orig.split([split_ratio], shuffle=None)
        
        self.metric_orig_train = BinaryLabelDatasetMetric(self.dataset_orig_train, 
                                             unprivileged_groups=self.unprivileged_groups,
                                             privileged_groups=self.privileged_groups)
        self.metric_orig_test = BinaryLabelDatasetMetric(self.dataset_orig_test, 
                                             unprivileged_groups=self.unprivileged_groups,
                                             privileged_groups=self.privileged_groups)
        
        # print('SPD', round(self.metric_orig_train.mean_difference(), 2))
        # print('DI', round(self.metric_orig_train.disparate_impact(), 2))
    

    def get_data_info(self):
        """get the information of the entire dataset

            returns:
                {'raw_data_num': , 'raw_data_features': []}
        """
        feature_lst = self.df.columns.values.tolist()
        try:
            feature_lst.remove('response')
        except:
            print('no this feature')

        return {
            'raw_data_num': self.df.shape[0],
            'raw_data_features': feature_lst
        }


    def get_table_data(self):
        """render table data into a tabel on a web page

        returns:
            {"rowData": [[], [], []...], "features": ['f1', 'f2', ...]}
            {
                columns: [a, b, c, d],
                data: [[], [], [], ...]
            }
        """
        dataset = self.df.copy()
        if self.is_bank_data:
            dataset = dataset[['age', 'job', 'marital', 'education', 'default', 
            'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'y']]

        rowData = dataset.values.tolist()
        if len(rowData) > 400:
            rowData = rowData[:400]
        features = dataset.columns.values.tolist()
        table_data = {'columns': features, 'data': rowData}
        return table_data
    

    def get_input_data_info(self):
        """get the information of the input data (train and test)
        for rendering the train & test panel

        returns:
            {'train': [[male_positive, male_negative], [female_positive, female_negative]], # number
            'test': [[male_positive, male_negative], [female_positive, female_negative]], # number      
            'attrVs': ['Male', 'Female']  # sensitive value }
        """
        def get_info(metric_data):
            print('positive ratio:', metric_data.num_positives(True)/(metric_data.num_positives(True)+metric_data.num_negatives(True)))
            print('positive ratio:', metric_data.num_positives(False)/(metric_data.num_positives(False)+metric_data.num_negatives(False)))
            return [[metric_data.num_positives(True), metric_data.num_negatives(True)], 
                    [metric_data.num_positives(False), metric_data.num_negatives(False)]]

        info = {}
        info['train'] = get_info(self.metric_orig_train)
        info['test'] = get_info(self.metric_orig_test)
        info['attrVs'] = ['Male', 'Female']

        return info
    
    
    def get_train_fair_metrics(self):
        """get the fair metric of the original an test data 
        """
        
        return ''
    
    def get_test_fair_metrics(self, type):
        """get the fair metric of the prediction data

        Args:
            'type': 'Original'/'Reweighing'/'LFR'/'OptimPreproc'
        Returns:
            {'SPD': [{'Original': 07}, {} ....], 'DI': [], 'EOD': [], 'AOD': []} 
            if this is the bank dataset
            [{'trainName': , 'modelName': , 'SPD': , 'DI': , 'EOD':, 'AOD':}, ...]
        """
        if self.is_bank_data:
            fair_metrics = []
            for idx, val in enumerate(self.bank_fair_names):
                fair_metric = {}
                name_lst = val.split(';')
                fair_metric['trainName'] = name_lst[0]
                fair_metric['modelName'] = name_lst[1]
                fair_metric['SPD'] = self.bank_fair_metrics[idx]['SPD']
                fair_metric['DI'] = self.bank_fair_metrics[idx]['DI']
                fair_metric['EOD'] = self.bank_fair_metrics[idx]['EOD']
                fair_metric['AOD'] = self.bank_fair_metrics[idx]['AOD']
                fair_metrics.append(fair_metric)
            return fair_metrics

        fair_metrics = {'SPD': [], 'DI': [], 'EOD': [], 'AOD': []}
        fair_metrics['SPD'].append({'Original': self.test_fair_metrics['SPD']['Original']})
        fair_metrics['DI'].append({'Original': self.test_fair_metrics['DI']['Original']})
        fair_metrics['EOD'].append({'Original': self.test_fair_metrics['EOD']['Original']})
        fair_metrics['AOD'].append({'Original': self.test_fair_metrics['AOD']['Original']})
        
        if type != 'Original':
            fair_metrics['SPD'].append({type: self.test_fair_metrics['SPD'][type]})
            fair_metrics['DI'].append({type: self.test_fair_metrics['DI'][type]})
            fair_metrics['EOD'].append({type: self.test_fair_metrics['EOD'][type]})
            fair_metrics['AOD'].append({type: self.test_fair_metrics['AOD'][type]})

        print('the return data is', fair_metrics)
        # add accuracy

        return fair_metrics
    
    def get_accuracies(self, type):
        """get the accuracies for the prediction data

        Args:
            'type': 'Original'/'Reweighing'/'LFR'/'OptimPreproc'
        Returns:
           [{'Original': 07}, {}]
           if this is the bank dataset
            [{'trainName': , 'modelName': , 'acc': }, ...]
        """
        if self.is_bank_data:
            accuracies = []
            for idx, val in enumerate(self.bank_fair_names):
                accuracy = {}
                name_lst = val.split(';')
                accuracy['trainName'] = name_lst[0]
                accuracy['modelName'] = name_lst[1]
                accuracy['acc'] = self.bank_fair_accuracies[idx]
                accuracies.append(accuracy)
            if len(accuracies) == 0:
                accuracies = [{'trainName': 'Original', 'modelName': 'LR', 'acc': 0.81}]
            return accuracies

        accuracies = []
        if 'Original' in self.test_accuracies:
            accuracies.append({'Original': self.test_accuracies['Original']})
        
        if type != 'Original':
            if type in  self.test_accuracies:
                accuracies.append({type: self.test_accuracies[type]})
        return accuracies

    def reweighing(self):
        """reweigh the original train data
        
            returns:
                (weights) [[male_p, male_n], [female_p, female_n]]
        """
        dataset_RW_train = ''
        if self.is_bank_data:
            dataset_RW_train_df = pd.read_csv(path.join(APP_STATIC,'uploads/preProcessRes/reweighing.csv'))
            dataset_RW_weights = np.loadtxt(path.join(APP_STATIC,'uploads/preProcessRes/weights.csv'))
            dataset_RW_train = BinaryLabelDataset(
                df=dataset_RW_train_df,
                label_names=['y'],  
                favorable_label=1, 
                unfavorable_label=0, 
                protected_attribute_names=['marital']
            )
            dataset_RW_train.instance_weights = dataset_RW_weights
            print(dataset_RW_train.instance_weights)
        else:
            RW = Reweighing(unprivileged_groups=self.unprivileged_groups,
                    privileged_groups=self.privileged_groups)
            dataset_RW_train = RW.fit_transform(self.dataset_orig_train)

        metric_RW_train = BinaryLabelDatasetMetric(dataset_RW_train, 
                                             unprivileged_groups=self.unprivileged_groups,
                                             privileged_groups=self.privileged_groups)

        self.miti_train_lst['Reweighing'] = dataset_RW_train
        print('reweighing')
        print('self.miti_train_lst', self.miti_train_lst)
        self.metric_miti_train_lst['Reweighing'] = metric_RW_train

        if self.is_bank_data:
            return [[0.76, 1.19], [1.47, 0.86]]
        else:
            return [[round(RW.w_p_fav, 2), round(RW.w_p_unfav, 2)], [round(RW.w_up_fav, 2), round(RW.w_up_unfav, 2)]]
    
    def LFR(self):
        """preprocess the train data
            Returns:
                num [[male_p, male_n], [female_p, female_n]]
        """
        flag = True
        dataset_LFR_train = ''
        metric_LFR_train = ''
        LFR_model = ''
        if self.is_bank_data:
            dataset_LFR_train_df = pd.read_csv(path.join(APP_STATIC,'uploads/preProcessRes/LFR.csv'))
            dataset_LFR_train = BinaryLabelDataset(
                df=dataset_LFR_train_df,
                label_names=['y'],  
                favorable_label=1, 
                unfavorable_label=0, 
                protected_attribute_names=['marital']
            )

            dataset_LFR_test_df = pd.read_csv(path.join(APP_STATIC,'uploads/preProcessRes/LFRTest.csv'))
            dataset_LFR_test = BinaryLabelDataset(
                df=dataset_LFR_test_df,
                label_names=['y'],  
                favorable_label=1, 
                unfavorable_label=0, 
                protected_attribute_names=['marital']
            )
            self.dataset_LFR_test = dataset_LFR_test
        else:
            while flag:
                LFR_model = LFR(unprivileged_groups=self.unprivileged_groups, 
                    privileged_groups=self.privileged_groups,
                    verbose=0, seed=9)
                LFR_model = LFR_model.fit(self.dataset_orig_train)
                dataset_LFR_train = LFR_model.transform(self.dataset_orig_train)
                self.dataset_LFR_test = LFR_model.transform(self.dataset_orig_test)
                metric_LFR_train = BinaryLabelDatasetMetric(dataset_LFR_train, 
                                                        unprivileged_groups=self.unprivileged_groups,
                                                        privileged_groups=self.privileged_groups)
                if metric_LFR_train.num_positives(True) != 0 and metric_LFR_train.num_negatives(True) != 0 and metric_LFR_train.num_positives(False)!=0 and metric_LFR_train.num_negatives(False)!=0:
                    flag = False
                print('one time')
                # save the test data
                # self.dataset_LFR_test.convert_to_dataframe()[0].to_csv('data/preProcessRes/LFRTest.csv', index=False)

                
        metric_LFR_train = BinaryLabelDatasetMetric(dataset_LFR_train, 
                                                    unprivileged_groups=self.unprivileged_groups,
                                                    privileged_groups=self.privileged_groups)
        metric_LFR_test = BinaryLabelDatasetMetric(self.dataset_orig_test, 
                                                    unprivileged_groups=self.unprivileged_groups,
                                                    privileged_groups=self.privileged_groups)
                
        self.miti_train_lst['LFR'] = dataset_LFR_train
        self.metric_miti_train_lst['LFR'] = metric_LFR_train
        return {'train': [[metric_LFR_train.num_positives(True), metric_LFR_train.num_negatives(True)], 
                    [metric_LFR_train.num_positives(False), metric_LFR_train.num_negatives(False)]],
                'test': [[metric_LFR_test.num_positives(True), metric_LFR_test.num_negatives(True)], 
                    [metric_LFR_test.num_positives(False), metric_LFR_test.num_negatives(False)]]}

    def optim_preproc(self):
        """preprocess the train data with the optim preproc technique

            Returns:
                num {train: [[male_p, male_n], [female_p, female_n]], test: ...}
        """
        dataset_optimPreproc_train = ''

        if self.is_bank_data:
            dataset_optimPreproc_train_df = pd.read_csv(path.join(APP_STATIC,'uploads/preProcessRes/optimPreprocTrain.csv'))
            dataset_optimPreproc_train = BinaryLabelDataset(
                df=dataset_optimPreproc_train_df,
                label_names=['y'],  
                favorable_label=1, 
                unfavorable_label=0, 
                protected_attribute_names=['marital']
            )
            dataset_optimPreproc_test_df = pd.read_csv(path.join(APP_STATIC,'uploads/preProcessRes/optimPreprocTest.csv'))
            self.dataset_optimPreproc_test = BinaryLabelDataset(
                df=dataset_optimPreproc_test_df,
                label_names=['y'],  
                favorable_label=1, 
                unfavorable_label=0, 
                protected_attribute_names=['marital']
            )
        else:
            new_df = copy.deepcopy(self.df)
            dataset_optimPreproc = load_preproc_data_bank() if self.is_bank_data else load_preproc_data(new_df)
            distortion_fun = get_distortion_bank if self.is_bank_data else get_distortion

            # test and train data under optim_preproc
            dataset_optimPreproc_train, dataset_optimPreproc_test = dataset_optimPreproc.split([self.split_ratio], shuffle=None)
            optim_options = {
                "distortion_fun": distortion_fun,
                "epsilon": 0.05,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }
            OptimPreproc_model = OptimPreproc(OptTools, optim_options, seed=1)
            print('init OptimPreproc_model finished')
            OptimPreproc_model = OptimPreproc_model.fit(dataset_optimPreproc_train)
            print('fit OptimPreproc_model finished')

            dataset_optimPreproc_train = OptimPreproc_model.transform(dataset_optimPreproc_train, transform_Y=True)
            
            # transform the test data also
            self.dataset_optimPreproc_test = OptimPreproc_model.transform(dataset_optimPreproc_test, transform_Y = True)
            # dataset_optimPreproc_train.convert_to_dataframe()[0].to_csv('data/preProcessRes/optimPreprocTrain.csv', sep=',', index=False)
            # self.dataset_optimPreproc_test.convert_to_dataframe()[0].to_csv('data/preProcessRes/optimPreprocTest.csv', sep=',', index=False)
        
        metric_optimPreproc_train = BinaryLabelDatasetMetric(dataset_optimPreproc_train, 
                                                    unprivileged_groups=self.unprivileged_groups,
                                                    privileged_groups=self.privileged_groups)
        metric_optimPreproc_test = BinaryLabelDatasetMetric(self.dataset_optimPreproc_test, 
                                                    unprivileged_groups=self.unprivileged_groups,
                                                    privileged_groups=self.privileged_groups)
        self.miti_train_lst['OptimPreproc'] = dataset_optimPreproc_train
        self.metric_miti_train_lst['OptimPreproc'] = metric_optimPreproc_train

        return {'train': [[metric_optimPreproc_train.num_positives(True), metric_optimPreproc_train.num_negatives(True)], 
                    [metric_optimPreproc_train.num_positives(False), metric_optimPreproc_train.num_negatives(False)]],
                    'test': [[metric_optimPreproc_test.num_positives(True), metric_optimPreproc_test.num_negatives(True)], 
                    [metric_optimPreproc_test.num_positives(False), metric_optimPreproc_test.num_negatives(False)]]}

    def preProcess(self, type):
        """
        calcualte different preporcessed train data

        Args:
            type (str): 'Reweighing' / 'LFR' / 'OptimPreproc'
        Returns:
            [[male_p, male_n], [female_p, female_n]] (when type = 'Reweighing', this is weights, otherwise, number)
            type=OptimPreproc: {'train': [[male_p, male_n], [female_p, female_n]], 'test': [[male_p, male_n], [female_p, female_n]]}
        """
        if type == 'Reweighing':
            return self.reweighing()
        elif type == 'LFR':
            return self.LFR()
        elif type == 'OptimPreproc':
            return self.optim_preproc()

    def train_model(self, model_name, train_name):
        """train the a model according to the model name using the specified train_data

        Args:
            model_name (str): 'LR': Logistic Regression; 'PrejudiceRmv': Prejudice Remover; 'Adversarial': Adversarial Debiasing
            train_name (str): 'Original': the original train data; 'Reweighing': the data after reweighing, 'OptimPreproc'
        """
        self.model_name = model_name
        self.train_name = train_name

        train_data = ''
        if train_name == 'Original':
            train_data = self.dataset_orig_train
        elif train_name == 'Reweighing':
            print(self.miti_train_lst)
            train_data = self.miti_train_lst[train_name]
            print('train the reweighing model')
        elif train_name == 'LFR':
            train_data = self.miti_train_lst[train_name]
        elif train_name == 'OptimPreproc':
            train_data = self.miti_train_lst[train_name]
        
        if model_name == 'LR':
            self.aif360Model = False
            self.model = LogisticRegression(random_state=0)
            if train_name == 'Reweighing':
                print(train_data.instance_weights)
                fit_params = {'sample_weight': train_data.instance_weights}
                self.model.fit(train_data.features, train_data.labels.ravel(), **fit_params)
            else:
                self.model.fit(train_data.features, train_data.labels.ravel())
            
            id = 'Original'
            if train_name == 'OptimPreproc':
                id = 'ptimPreproc_Original'
            # info = self.test(id, '')
        elif model_name == 'Adversarial':
            # adjust the parameters under different cases
            self.aif360Model = True
            sess = tf.Session()
            num_epochs = 50
            classifier_num_hidden_units = 200
            
            if not self.is_bank_data:
                num_epochs = 200
                classifier_num_hidden_units = 80
            elif train_name == 'LFR':
                num_epochs = 30
                classifier_num_hidden_units = 100

            self.model = AdversarialDebiasing(privileged_groups = self.privileged_groups,
                                unprivileged_groups = self.unprivileged_groups,
                                scope_name='debiased_classifier',
                                debias=True,
                                sess=sess,
                                classifier_num_hidden_units=classifier_num_hidden_units,
                                num_epochs=num_epochs,
                                seed=0)
            
            self.model.fit(train_data)
            # info = self.test('Original', '')
            tf.reset_default_graph()

            # if this is Adversarial debiasing, the store the test result first
            self.AdversarialTest = self.test('test', train_name, model_name, req_from_inner=True)
            print('self.AdversarialTest', self.AdversarialTest)
            self.model = ''
           
        elif model_name == 'PrejudiceRmv':
            self.aif360Model = True
            self.model = PrejudiceRemover(eta=0.1)
            self.model.fit(train_data)
          

        return {}

    def test(self, data_name, name='', model = '', req_from_inner = False):
        """test/predict on the trained model (we always use the latest model)

            Args:
                data_name (str): 'Original': the original train data; 'Reweighing': the data after reweighing, 'test': 
                name(str): 'Original', 'Reweighing', 'LFR' ... 'Adversarial' ..''. the name for the fairness metrics, if name is null, don't compute the fairness metrics
                model(str): the name of model
            Returns:
            {
                'data': [ [[true_positive, False_positive], [False_negative, true_negative]],
                [[true_positive, False_positive], [False_negative, true_negative]] ],  # confusion matrix for the two groups

                'accuracy': 0.98,

                'attrVs': ['Male', 'Female']  # sensitive value
            }
        """
        data_info = {}
        data = ''
        if data_name == 'Original':
            data = self.dataset_orig_train
        elif data_name == 'test':
            data = self.dataset_orig_test
        elif data_name == 'ptimPreproc_Original':
            data = self.miti_train_lst['OptimPreproc']

        if name == 'OptimPreproc':
            data = self.dataset_optimPreproc_test
        # elif name == 'LFR':
        #     data = self.dataset_LFR_test
        #     print('the test data is LFR')
        dataset_pred = ''

        # if the test model is  Adversarial
        if model == 'Adversarial' and not req_from_inner:
            return self.AdversarialTest

        if self.aif360Model:
            dataset_pred = self.model.predict(data)
        else:
            y_val_pred = self.model.predict(data.features)
            # print(np.transpose([y_val_pred]))
            dataset_pred = data.copy()
            # print(dataset_pred.labels)
            dataset_pred.labels = np.transpose([y_val_pred])
            # print(dataset_pred.scores)
            if name == 'Original' and not self.is_bank_data:
                # get the probability of label = 1
                # predict_proba(data)[:,1]
                scores = np.transpose([self.model.predict_proba(data.features)[:,1]])
                dataset_pred.scores = scores
                # print('res', self.model.predict(data.features))
                self.dataset_orig_pred = dataset_pred

        metric = ClassificationMetric(    
                    data, dataset_pred,
                    unprivileged_groups=self.unprivileged_groups,
                    privileged_groups=self.privileged_groups)
        p_confuison_matrix = metric.binary_confusion_matrix(True)
        np_confuison_matrix = metric.binary_confusion_matrix(False)

        data_info['data'] = [[[p_confuison_matrix['TP'], p_confuison_matrix['FN']], [p_confuison_matrix['FP'], p_confuison_matrix['TN']]], 
                            [[np_confuison_matrix['TP'], np_confuison_matrix['FN']], [np_confuison_matrix['FP'], np_confuison_matrix['TN']]]]

        accuracy = metric.accuracy()
        data_info['accuracy'] = accuracy
        data_info['attrVs'] = ['Male', 'Female']
        # data_info['attrVs'] = ['unmarried', 'married']

        # set the fair metrics
        if name:
            SPD = round(metric.statistical_parity_difference(), 2)
            DI = round(metric.disparate_impact(), 2)
            EOD = round(metric.equal_opportunity_difference(), 2)
            AOD = round(metric.average_odds_difference(), 2)
            
            SPD = 100 if pd.isna(float(SPD)) else SPD
            DI = 100 if pd.isna(float(DI)) else DI
            EOD = 100 if pd.isna(float(EOD)) else EOD
            AOD = 100 if pd.isna(float(AOD)) else AOD

            if self.is_bank_data:
                # store the accuarcy and the fair metrics into the bank
                fair_name = str(self.train_name)+';'+str(self.model_name)
                try:
                    # if this name already exists
                    idx = self.bank_fair_names.index(fair_name)
                    del self.bank_fair_names[idx]
                    del self.bank_fair_metrics[idx]
                    del self.bank_fair_accuracies[idx]
                except:
                    pass
                finally:
                    # append this
                    self.bank_fair_names.append(fair_name)
                    self.bank_fair_accuracies.append(round(accuracy, 2))
                    self.bank_fair_metrics.append({'SPD': SPD, 'DI': DI, 'EOD': EOD, 'AOD':AOD})
                print('current bank_fair_names', self.bank_fair_names)
                print('current bank_fair_accuracies', self.bank_fair_accuracies)
                print('current bank_fair_metrics', self.bank_fair_metrics)
            else:
                # get the fair metric of this original train
                if model!='LR':
                    name = model
                self.test_fair_metrics['SPD'][name] = SPD
                self.test_fair_metrics['DI'][name] = DI
                self.test_fair_metrics['EOD'][name] = EOD
                self.test_fair_metrics['AOD'][name] = AOD
                self.test_accuracies[name] = round(accuracy, 2)
        return data_info
    

    def ROC(self):
        """post processing: RejectOptionClassification 
        """
        data_info = {}
        roc_model = RejectOptionClassification(privileged_groups = self.privileged_groups,
                                unprivileged_groups = self.unprivileged_groups, num_class_thresh=500)
        roc_model = roc_model.fit(self.dataset_orig_test, self.dataset_orig_pred)
        post_res = roc_model.predict(self.dataset_orig_pred)

        metric = ClassificationMetric(    
                    self.dataset_orig_test, post_res,
                    unprivileged_groups=self.unprivileged_groups,
                    privileged_groups=self.privileged_groups)
        p_confuison_matrix = metric.binary_confusion_matrix(True)
        np_confuison_matrix = metric.binary_confusion_matrix(False)

        accuracy = metric.accuracy()
        data_info['accuracy'] = accuracy
        data_info['attrVs'] = ['Male', 'Female']

        data_info['data'] = [[[p_confuison_matrix['TP'], p_confuison_matrix['FN']], [p_confuison_matrix['FP'], p_confuison_matrix['TN']]], 
                            [[np_confuison_matrix['TP'], np_confuison_matrix['FN']], [np_confuison_matrix['FP'], np_confuison_matrix['TN']]]]
        SPD = round(metric.statistical_parity_difference(), 2)
        DI = round(metric.disparate_impact(), 2)
        EOD = round(metric.equal_opportunity_difference(), 2)
        AOD = round(metric.average_odds_difference(), 2)
        self.test_fair_metrics['SPD']['ROC'] = SPD
        self.test_fair_metrics['DI']['ROC'] = DI
        self.test_fair_metrics['EOD']['ROC'] = EOD
        self.test_fair_metrics['AOD']['ROC'] = AOD
        self.test_accuracies['ROC'] = round(accuracy, 2)

        return data_info


if __name__ == '__main__':
    # /Users/yanxinyuan/Desktop/FairAI/fairVenv/lib/python3.8/site-packages/aif360/data/raw/bank/bank-additional-full.csv
    # fair_AI = fairAI('data/bank-additional-full.csv', is_bank_data=True)
    # fair_AI = fairAI('data/creditScore.csv')
    # fair_AI.customize_data(split_ratio=0.7, protected_attribute_names=['gender'])
    # print(fair_AI.preProcess('LFR'))

    fairAIObj = fairAI(path.join(APP_STATIC,'uploads/creditScore.csv'))
    fairAIObj.customize_data(split_ratio=0.7)
    fairAIObj.train_model('PrejudiceRmv', 'Original')   # train the Logistic Regression model
    print(fairAIObj.test('test', 'PrejudiceRmv', 'PrejudiceRmv'))  # evaluate the model with the test data
    # print(fairAIObj.ROC())


    # print(fair_AI.train_model('PrejudiceRmv', 'LFR'))
    # print('finish')
    # print('test result', fair_AI.test('test', 'LFR', 'PrejudiceRmv'))

    # print('The info of train and test data:', fair_AI.get_input_data_info())
    # fair_AI.LFR()

    # for i in range(5):
    #     print(fair_AI.train_model('Adversarial', 'Original'))
    #     print('test result', fair_AI.test('test', 'Original'))

    # print(fair_AI.test('test', 'original'))
    # print(fair_AI.get_input_data_info())

    # fair_AI.get_table_data()
