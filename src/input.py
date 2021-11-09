import pandas as pd

# datas
train = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
test = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')
ss = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')

pd.set_option('display.max_columns', None)