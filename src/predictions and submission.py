import pandas as pd
from config import ss
from model_1 import model_1_preds
from model_2 import model_2_preds
from model_3 import model_3_preds
from model_4 import model_4_preds
from model_5 import model_5_preds
from model_6 import model_6_preds

prediction = pd.DataFrame()
prediction['preds1'] = model_1_preds
prediction['preds2'] = model_2_preds
prediction['preds3'] = model_3_preds
prediction['preds4'] = model_4_preds
prediction['preds5'] = model_5_preds
prediction['preds6'] = model_6_preds
prediction['preds'] = (prediction.preds1 + prediction.preds2 + prediction.preds3 + prediction.preds4 + prediction.preds5 + prediction.preds6)/6

ss['pressure'] = prediction.preds
ss.to_csv('Submission with blend.csv', index = False)
ss.head()