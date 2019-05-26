import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

# Data Study
source = pd.read_csv('./data/organized_data.csv')
pd.set_option('display.max_columns', 18)

# Data_Pre Processing

#to_drop = ['ab_id', 'ax', 'ay', 'az', 'break_angle', 'break_length', 'break_y', 'code', 'end_speed', 'nasty', 'pitch_num', 'px', 'pz', 'spin_dir', 'sz_bot', 'sz_top', 'type_confidence', 'vx0', 'vy0', 'vz0', 'x', 'x0', 'y', 'y0', 'z0', 'zone']
#to_drop = ['ab_id', 'code', 'sz_bot', 'sz_top', 'type_confidence', 'vx0', 'vy0', 'vz0', 'x', 'x0', 'y', 'y0', 'z0', 'zone']
#source.drop(to_drop, inplace=True, axis=1)

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
encoder = LabelEncoder()
source['pitch_type'] = encoder.fit_transform(source['pitch_type'])
#mapping = dict(zip(encoder.classes_, range(1, len(encoder.classes_)+1)))
#print('Mapping of Pitch_type " ', mapping)
# CH': 0, 'CU': 1, 'FC': 2, 'FF': 3, 'FS': 4, 'FT': 5, 'KC': 6, 'KN': 7, 'SI': 8, 'SL': 9

# Test Train Split
features = ['b_count', 's_count', 'outs', 'on_1b', 'on_2b', 'on_3b']

X = source[features]
y = source.pitch_type

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Decision Tree Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

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
sample_array = {'b_count' : [2], 's_count' : [2], 'outs' : [2], 'on_1b' : [1], 'on_2b' : [0], 'on_3b' : [0]}
test_sample = pd.DataFrame(data=sample_array)
special_pred = rfc.predict(test_sample)

if (special_pred == 0) :
    print('Changeup')
if (special_pred == 1) :
    print('Curveball')
if (special_pred == 2) :
    print('Cutter')
if (special_pred == 3) :
    print('Four-seam Fastball')
if (special_pred == 4) :
    print('Splitter')
if (special_pred == 5) :
    print('Two-seam Fastball')
if (special_pred == 6) :
    print('Knuckle curve')
if (special_pred == 7) :
    print('Knuckeball')
if (special_pred == 8) :
    print('Sinker')
if (special_pred == 9) :
    print('Slider')

for i in range(len(source)):
    if (source.pitch_type[i] == special_pred) :
        x = source.px[i]
        y = source.pz[i]
        plt.title("Pitch location")
        plt.scatter(x, y, c='r')
plt.show()