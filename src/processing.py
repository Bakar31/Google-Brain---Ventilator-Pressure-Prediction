from sklearn.preprocessing import StandardScaler
from input import train, test

targets = train[['pressure']].to_numpy().reshape(-1, 80)
train.drop(['pressure', 'id', 'breath_id'], axis = 1, inplace = True)
test = test.drop(['id', 'breath_id'], axis = 1)
print('Step-1 done')

SC = StandardScaler()
train = SC.fit_transform(train)
test = SC.transform(test)
print('Step-2 done')

train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, train.shape[-1])
print('Step-3 done')

idx_len = round(0.90*len(train))
X_train, X_valid = train[0:idx_len], train[idx_len:]
y_train, y_valid = targets[0:idx_len], targets[idx_len:]
print('Step-4 done')