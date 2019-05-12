import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

def pitch_type_mapping(type_code):
    return {'FF': 'b',
            'SL': 'g',
            'SI': 'r',
            'FT': 'c',
            'CH': 'm',
            'CU': 'y',
            'FC': 'k',
            'FS': 'w',
            'KC': 'brown',
            'FA': 'coral'}[type_code]

if __name__ == "__main__":
    # import the raw data
    raw_pitch_df = pd.read_csv('./data/pitches000001.csv')
    raw_bat_df = pd.read_csv('./data/atbats.csv')
    # print(raw_pitch_df)

    # remove novelty pitches and non-standard pitch events like pitchouts
    raw_pitch_df = raw_pitch_df[raw_pitch_df.pitch_type.isin(['FF', 'SL', 'SI', 'FT', 'CH', 'CU', 'FC', 'FS', 'KC', 'FA'])]

    # fetch batter data which was out
    raw_bat_df = raw_bat_df[raw_bat_df.event.isin([
        'Bunt Groundout', 'Bunt Lineout', 'Bunt Pop Out', 'Double Play',
        'Fielders Choice Out', 'Flyout', 'Forceout', 'Grounded Into DP',
        'Groundout', 'Lineout', 'Pop Out', 'Sac Fly DP', 'Sacrifice Bunt DP',
        'Strikeout', 'Strikeout - DP', 'Triple Play'])]

    x = []
    z = []
    t = []

    for i in range(1000):


        to_list = raw_bat_df['ab_id'].values.tolist()

        arranged_data = raw_pitch_df[raw_pitch_df.ab_id.isin([to_list[i]])]
        arranged_data = arranged_data.loc[:, ["px", "pz", "ab_id", "pitch_type", "pitch_num"]]
        index = arranged_data.groupby(['ab_id'])['pitch_num'].transform(max) == arranged_data['pitch_num']


        x.append(float(arranged_data.px[index]))
        z.append(float(arranged_data.pz[index]))
        t.append(arranged_data.iloc[len(index) - 1]['pitch_type'])

        #x = float(arranged_data.px[index])
        #z = float(arranged_data.pz[index])
        #t = arranged_data.iloc[len(index) - 1]['pitch_type']

        #print("x:",x)
        #print("z:",z)
        #print("t:",t)

    for i in range(len(t)):
        t[i] = pitch_type_mapping(t[i])

    #print(t)

    plt.figure()
    plt.scatter(x, z, c=t)
    plt.show()



