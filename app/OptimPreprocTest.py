import math
from re import A
import sys
from app import APP_STATIC
from os import path

# from fairer.fair import ARTClassifier_model
# sys.path.insert(1, "../")  

import numpy as np
np.random.seed(0)
import pandas as pd

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import LFR
from optim_preproc_helpers.optim_preproc import OptimPreproc
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.datasets import StandardDataset
from optim_preproc_helpers.opt_tools import OptTools
from aif360.datasets import BankDataset

# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()

from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from optim_preproc_helpers.data_preproc_functions import load_preproc_data_bank
from optim_preproc_helpers.distortion_functions import get_distortion_bank

def test(dataset, model, fair360 = False):
    metric = ''

    if fair360:
        dataset_pred = model.predict(dataset)
        metric = ClassificationMetric(
                    dataset, dataset_pred,
                    unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    else:
        y_val_pred = model.predict(dataset.features)
        # calculate the confusion matrix
        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                    dataset, dataset_pred,
                    unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
   
    info = {
        'accuracy': metric.accuracy(),
        'confusionMatrix': metric.binary_confusion_matrix(),
        'disparate_impact': metric.disparate_impact(),
        'EOD': metric.equal_opportunity_difference(),
        'AOD': metric.average_odds_difference(),
        'SPD': metric.statistical_parity_difference()
    }
    print('info', info)


def load_preproc_data(df):
    """
    process the data into categorical attributes
        - dependents=1, dependents=2, ageGroup1 
    Args:
        df : dataframe
    Returns:
        midified dataframe
    """
    def custom_preprocessing(df):
        """
            continous -> categorical (replace the original values)
            dependents; age; amount
        """
        def group_age(x):
            '''
            19-25(0); 25-30(1); 30-35(2); 35-40(3); 40-50(4); 50-60(5); >60(6);
            '''
            x = int(x)
            if x<25:
                return 0
            elif 25<=x<30:
                return 1
            elif 30<=x<40:
                return 2
            elif 40<=x<55:
                return 3
            else:
                return 4

        def group_amount(x):
            '''
            250-500(0)500-1000(1)1000-1500(2)1500-2000(3)2000-3000(4)3000-4000(5)4000-5000(6)>5000(7)
            '''
            x = int(x)
            if x<1000:
                return 0
            elif 1000<=x<2000:
                return 1
            elif 2000<=x<3000:
                return 2
            elif 3000<=x<5000:
                return 3
            else:
                return 4

        df['age'] = df['age'].apply(lambda x: group_age(x))
        df['amount'] = df['amount'].apply(lambda x: group_amount(x))

        return df

    return StandardDataset(df, 
        label_name='response', 
        favorable_classes=[1],
        protected_attribute_names=['gender'],
        privileged_classes=[[1]],
        instance_weights_name=None,
        categorical_features=['age', 'amount'],
        features_to_drop=[],
        custom_preprocessing=custom_preprocessing
        )


def get_distortion(vold, vnew):
    '''
    employment,dependents,age,amount,response
    '''
    # print('vold', vold)
    # print('vnew', vnew)
    def adjust(a):
        return int(a)
    
    def getcost(attr, new, old):
        new = adjust(new)
        old = adjust(old)
        dis = abs(new-old)

        if attr == 'employment':
            return 2 if dis == 1 else 0
        elif attr == 'dependents':
            return 1 if dis == 1 else 0
        elif attr == 'age':
            if dis == 0:
                return 0
            elif dis < 2:
                return 1
            elif dis < 4:
                return 2
            else:
                return 3
        elif attr == 'amount':
            if dis == 0:
                return 0
            elif dis < 2:
                return 1
            elif dis < 4:
                return 2
            else:
                return 3
        elif attr == 'response':
            return 1 if dis == 1 else 0

    total_cost = 0.0
    for k in vold:
        if k in vnew:
             total_cost += getcost(k, vnew[k], vold[k])
    return total_cost


def print_info(metric):
    print(metric.num_negatives(False))
    print(metric.num_positives(False))
    print(metric.num_negatives(True))
    print(metric.num_positives(True))
    print(metric.num_instances(False))

    # measure the fair metric 
    metric_orig_train_SPD = metric.mean_difference()
    metric_orig_train_DI = metric.disparate_impact()
    print('metric_orig_train_SPD', metric_orig_train_SPD)
    print('metric_orig_train_DI', metric_orig_train_DI)


if __name__ == '__main__':
    # df = pd.read_csv('data/creditScore.csv')
    # privileged_groups = [{'gender': 1}]
    # unprivileged_groups = [{'gender': 0}]

    # dataset_orig = load_preproc_data(df)

    # dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

    # metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
    #                                          unprivileged_groups=unprivileged_groups,
    #                                          privileged_groups=privileged_groups)
    # print_info(metric_orig_train)
    
    # test for the bank data
    df = pd.read_csv(path.join(APP_STATIC, 'uploads/bank_5000.csv'), sep=';')    # raw dataset
    protected_attribute_maps = [{1.0: 'unmarried', 0.0: 'married'}]

    dataset_orig = load_preproc_data_bank()
    print(dataset_orig.convert_to_dataframe())

    privileged_groups = [{'marital': 1.0}]
    unprivileged_groups = [{'marital': 0.0}]
    print('after loading the dataset')

    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
    
    metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                            unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups)

    print_info(metric_orig_train)

    optim_options = {
        "distortion_fun": get_distortion_bank,
        "epsilon": 0.05,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.1, 0.05, 0]
    }
    OptimPreproc_model = OptimPreproc(OptTools, optim_options)
    print('init OptimPreproc_model finished')
    OptimPreproc_model = OptimPreproc_model.fit_transform(dataset_orig_train)
    print('fit OptimPreproc_model finished')

    # dataset_transf_train = OptimPreproc_model.transform(dataset_orig_train, transform_Y=True)
    # metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
    #                                            unprivileged_groups=unprivileged_groups,
    #                                            privileged_groups=privileged_groups)
    # print_info(metric_transf_train)




    # train with the original data
    # model = LogisticRegression(random_state=0)
    # fit_params = {'sample_weight': dataset_orig_train.instance_weights}
    # lr_orig_train = model.fit(dataset_orig_train.features, dataset_orig_train.labels.ravel(), **fit_params)
    # test(dataset_orig_test, lr_orig_train)

    # train with the modified model
    # model = LogisticRegression(random_state=0)
    # fit_params = {'sample_weight': dataset_transf_train.instance_weights}
    # lr_transf_train = model.fit(dataset_transf_train.features, dataset_transf_train.labels.ravel(), **fit_params)
    # dataset_transf_test = OptimPreproc_model.transform(dataset_orig_test, transform_Y = True)
    # test(dataset_transf_test, lr_transf_train)


