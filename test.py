import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping

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

dummy_y = pd.get_dummies(y)
dummy_y.shape

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
    model.fit(X, Y, epochs=5)
    return model


X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.2, random_state=1)

model = buildModel(X_train, y_train, 50)

