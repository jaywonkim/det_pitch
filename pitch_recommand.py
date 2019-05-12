import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # import the raw data
    raw_pitch_df = pd.read_csv('./data/pitches000000.csv')
    raw_bat_df = pd.read_csv('./data/pujols_data.csv')

    # remove novelty pitches and non-standard pitch events like pitchouts
    raw_pitch_df = raw_pitch_df[raw_pitch_df.pitch_type.isin(['FF', 'SL', 'SI', 'FT', 'CH', 'CU', 'FC', 'FS', 'KC', 'FA'])]

    to_list = raw_bat_df['ab_id'].values.tolist()
    base_data = pd.DataFrame()

    for i in range(len(raw_bat_df)):
        arranged_data = raw_pitch_df[raw_pitch_df.ab_id.isin([to_list[i]])]
        arranged_data = arranged_data.loc[:, ["px", "pz", "ab_id", "pitch_type", "b_count", "s_count", "outs", "pitch_num", "on_1b", "on_2b", "on_3b"]]

        arranged_data['event'] = str("N/A")
        index = arranged_data.groupby(['ab_id'])['pitch_num'].transform(max) == arranged_data['pitch_num']
        arranged_data = arranged_data.reset_index(drop=True)
        arranged_data.set_value(len(index)-1, 'event', raw_bat_df.event[i])

        base_data = base_data.append(arranged_data)

    base_data = base_data.reset_index(drop=True)
    print(base_data)

    # some feature creation/refinement
    base_data['is_out'] = np.where(base_data.event.isin([
        'Bunt Groundout', 'Bunt Lineout', 'Bunt Pop Out', 'Double Play',
        'Fielders Choice Out', 'Flyout', 'Forceout', 'Grounded Into DP',
        'Groundout', 'Lineout', 'Pop Out', 'Sac Fly DP', 'Sacrifice Bunt DP',
        'Strikeout', 'Strikeout - DP', 'Triple Play']), 1, 0)

    base_data['balls'] = np.where(~base_data.b_count.isnull(), 1, 0)
    base_data['strikes'] = np.where(~base_data.s_count.isnull(), 1, 0)
    base_data['outs'] = np.where(~base_data.outs.isnull(), 1, 0)

    base_data['runner_on_1st'] = np.where(~base_data.on_1b.isnull(), 1, 0)
    base_data['runner_on_2nd'] = np.where(~base_data.on_2b.isnull(), 1, 0)
    base_data['runner_on_3rd'] = np.where(~base_data.on_3b.isnull(), 1, 0)
    base_data['total_baserunners'] = base_data['runner_on_1st'] + base_data['runner_on_2nd'] + base_data['runner_on_3rd']


    test_df, train_df = train_test_split(base_data, test_size=0.2)

    # selecting features for training
    features = ['balls',
                'strikes',
                'outs',
                'runner_on_1st',
                'runner_on_2nd',
                'runner_on_3rd',
                'total_baserunners']

    response = 'is_out'

    # create and train random forest
    runForrest = RandomForestClassifier(n_estimators=200, n_jobs=1, max_depth=10, max_features=7, min_samples_leaf=10, min_samples_split=20)

    runForrest.fit(train_df[features], train_df[response])

    probas = runForrest.predict_proba(test_df[features])
    preds = runForrest.predict(test_df[features])

    # some model performance metrics
    print('AUC: ' + str(metrics.roc_auc_score(y_score=probas[:, 1], y_true=test_df[response])))
    print('logloss: ' + str(metrics.log_loss(y_pred=probas[:, 1], y_true=test_df[response])))
    print('accuracy score: ' + str(metrics.accuracy_score(test_df[response], preds)))
    print('null accuracy: ' + str(max(test_df[response].mean(), 1 - test_df[response].mean())))
    print(metrics.confusion_matrix(test_df[response], preds))

    # histogram of predicted probabilities
    plt.hist(probas[:, 1], bins=10)
    plt.xlim(0, 1)
    plt.title('Histogram of predicted probabilities')
    plt.xlabel('Predicted probability of fastball')
    plt.ylabel('Frequency')
    plt.show()

    fpr, tpr, thresholds = metrics.roc_curve(test_df[response], probas[:, 1])
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for fastball predictor')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.show()
    
