import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

# Data Study
source = pd.read_csv('./data/organized_data.csv')
pd.set_option('display.max_columns', 18)

# Fill Null values
source['pitch_type'].fillna("FF", inplace=True)
#source['px'].fillna(source['px'].mean(), inplace=True)
#source['pz'].fillna(source['pz'].mean(), inplace=True)
source['b_count'].fillna(source['b_count'].mean(), inplace=True)
source['s_count'].fillna(source['s_count'].mean(), inplace=True)
source['outs'].fillna(source['outs'].mean(), inplace=True)
source['pitch_num'].fillna(source['pitch_num'].mean(), inplace=True)
source['on_1b'].fillna(0, inplace=True)
source['on_2b'].fillna(0, inplace=True)
source['on_3b'].fillna(0, inplace=True)
source['event'].fillna("Strikeout", inplace=True)
source.dropna()

# Encode the result
#encoder = LabelEncoder()
#source['pitch_type'] = encoder.fit_transform(source['pitch_type'])
#mapping = dict(zip(encoder.classes_, range(0, len(encoder.classes_)+1)))
#print('Mapping of Pitch_type " ', mapping)
source['pitch_type'] = np.where(source.pitch_type.isin(['FC', 'FF', 'FT']), 1, 0)
#print(source['pitch_type'])

# Test Train Split
features = ['b_count', 's_count', 'outs', 'on_1b', 'on_2b', 'on_3b']

X = source[features]
y = source.pitch_type

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Decision Tree Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# View the decision tree
pydot = StringIO()
export_graphviz(clf, out_file=pydot, filled=True, rounded=True, special_characters=True,
                feature_names=features)
graph = pydotplus.graph_from_dot_data(pydot.getvalue())
graph.write_png('DecisionTree.png')
Image(graph.create_png())
print("Decision Tree Created")

# Random Forest Model
rfc = RandomForestClassifier(n_estimators=10)
rfc = rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
#print(y_pred)

print("Random Forest Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("훈련 세트 정확도 : {:.3f}".format(rfc.score(X_train,y_train)))
print("테스트 세트 정확도 : {:.3f}".format(rfc.score(X_test,y_test)))
print("특성 중요도 : \n{}".format(rfc.feature_importances_))

# 특성 중요도 시각화 하기
importances = rfc.feature_importances_

std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

plt.title("information gain")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

# 특정상황에서의 구질예측을 위한 test set 설정
set_b_count = 3
set_s_count = 2
set_outs = 2
set_on_1b = 1
set_on_2b = 1
set_on_3b = 1

sample_array = {'b_count' : [set_b_count], 's_count' : [set_s_count], 'outs' : [set_outs], 'on_1b' : [set_on_1b], 'on_2b' : [set_on_2b], 'on_3b' : [set_on_3b]}
test_sample = pd.DataFrame(data=sample_array)
special_pred = rfc.predict(test_sample)

print("Ball Count : ", set_b_count, "ball ", set_s_count, "strike ", set_outs, "out ")
print("Runner : ", set_on_1b, "on First ", set_on_2b, "on Second ", set_on_3b, "on Third ")

if (special_pred == 0) :
    print('Off Speed type')
if (special_pred == 1) :
    print('FastBall type')


for i in range(len(source)):
    if (source.pitch_type[i] == special_pred) :
        x = source.px[i]
        y = source.pz[i]
        plt.title("Pitch location")
        plt.scatter(x, y, c='r')
plt.show()

'''
# Dense Neural Network
X = source[features]
y = source.pitch_type
dummy_y = pd.get_dummies(y)

def buildModel(X, Y, hidden):
    inputs = X.shape[1]
    outputs = Y.shape[1]

    model = Sequential()
    model.add(Dense(hidden, input_dim=inputs, activation='relu'))  # variable inputs, 50 hidden neurons
    model.add(Dense(outputs, activation='softmax'))  # 15 possible pitch types

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit model
    print('start fit')
    model.fit(X, Y, epochs=50)

    return model

X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.1, random_state=1)
model = buildModel(X_train, y_train, 30)
model.summary()

# plot training history
# simple early stopping
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
#history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0, callbacks=[es])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)
plt.yscale('linear')
plt.axis([0,100,0,1])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
#special_pred_by_nn = model.predict(test_sample)
#print(special_pred_by_nn)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)
plt.yscale('linear')
plt.axis([0,100,0,1])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show()'''
