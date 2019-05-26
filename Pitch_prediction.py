import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

# Data Study
source = pd.read_csv('./data/pitches_test.csv')
pd.set_option('display.max_columns', 18)

# Data_Pre Processing

#to_drop = ['ab_id', 'ax', 'ay', 'az', 'break_angle', 'break_length', 'break_y', 'code', 'end_speed', 'nasty', 'pitch_num', 'px', 'pz', 'spin_dir', 'sz_bot', 'sz_top', 'type_confidence', 'vx0', 'vy0', 'vz0', 'x', 'x0', 'y', 'y0', 'z0', 'zone']
to_drop = ['ab_id', 'code', 'sz_bot', 'sz_top', 'type_confidence', 'vx0', 'vy0', 'vz0', 'x', 'x0', 'y', 'y0', 'z0', 'zone']
source.drop(to_drop, inplace=True, axis=1)

# Fill Null values
source['pitch_type'].fillna("FF", inplace=True)
source['type'].replace(to_replace="X", value="B", inplace=True)
source['pfx_x'].fillna(source['pfx_x'].mean(), inplace=True)
source['pfx_z'].fillna(source['pfx_z'].mean(), inplace=True)
source['spin_rate'].fillna(source['spin_rate'].mean(), inplace=True)
source['start_speed'].fillna(source['start_speed'].mean(), inplace=True)
source['ax'].fillna(source['ax'].mean(), inplace=True)
source['ay'].fillna(source['ay'].mean(), inplace=True)
source['az'].fillna(source['az'].mean(), inplace=True)
source['b_count'].fillna(source['b_count'].mean(), inplace=True)
source['b_score'].fillna(source['b_score'].mean(), inplace=True)
source['break_angle'].fillna(source['break_angle'].mean(), inplace=True)
source['b_count'].fillna(source['b_count'].mean(), inplace=True)
source['b_score'].fillna(source['b_score'].mean(), inplace=True)
source['break_length'].fillna(source['break_length'].mean(), inplace=True)
source['break_y'].fillna(source['break_y'].mean(), inplace=True)
source['end_speed'].fillna(source['end_speed'].mean(), inplace=True)
source['nasty'].fillna(source['nasty'].mean(), inplace=True)
source['pitch_num'].fillna(source['pitch_num'].mean(), inplace=True)
source['px'].fillna(source['px'].mean(), inplace=True)
source['pz'].fillna(source['pz'].mean(), inplace=True)
source['spin_dir'].fillna(source['spin_dir'].mean(), inplace=True)
source.dropna()

# Encode the result
encoder = LabelEncoder()
source['pitch_type'] = encoder.fit_transform(source['pitch_type'])
source['type'] = encoder.fit_transform(source['type'])

# Test Train Split
features = ['ax', 'ay', 'az', 'b_count', 'b_score', 'break_angle', 'b_count', 'b_score', 'break_length',
            'break_y', 'end_speed', 'nasty', 'pitch_num', 'px', 'pz', 'spin_dir', 'on_1b', 'on_2b',
            'on_3b', 'outs', 'pfx_x', 'pfx_z', 'type', 's_count', 'spin_rate', 'start_speed']
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

#for i in range(len(y_pred)):
#    print(y_pred[i])

print("Random Forest Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#probas = format(rfc.predict_proba(X_test))
probas = rfc.predict_proba(X_test)
#for i in range(len(probas)):
#    print(probas[i])
print(probas.shape)


print(y_test)

print(y_train.shape)
