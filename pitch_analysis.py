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

    # remove novelty pitches and non-standard pitch events like pitchouts
    raw_pitch_df = raw_pitch_df[raw_pitch_df.pitch_type.isin(['FF', 'SL', 'SI', 'FT', 'CH', 'CU', 'FC', 'FS', 'KC', 'FA'])]

    # some feature creation/refinement
    raw_pitch_df['is_fastball'] = np.where(raw_pitch_df.pitch_type.isin(['FF', 'SI', 'SI', 'FT', 'FA', 'FS']), 1, 0)
    raw_pitch_df['runner_on_1st'] = np.where(~raw_pitch_df.on_1b.isnull(), 1, 0)
    raw_pitch_df['runner_on_2nd'] = np.where(~raw_pitch_df.on_2b.isnull(), 1, 0)
    raw_pitch_df['runner_on_3rd'] = np.where(~raw_pitch_df.on_3b.isnull(), 1, 0)
    raw_pitch_df['total_baserunners'] = raw_pitch_df['runner_on_1st'] + raw_pitch_df['runner_on_2nd'] + raw_pitch_df['runner_on_3rd']
    raw_pitch_df['is_offspeed'] = 1 - raw_pitch_df['is_fastball']
    print(raw_pitch_df)
    test_df, train_df = train_test_split(raw_pitch_df, test_size=0.2)

    # selecting features for training
    features = [#'inning',
                #'top',
                #'at_bat_num',
                #'pcount_at_bat',
                #'pcount_pitcher',
                #'balls',
                #'strikes',
                #'fouls',
                'outs',
                'runner_on_1st',
                'runner_on_2nd',
                'runner_on_3rd',
                'total_baserunners']

    response = 'is_fastball'

    # create and train random forest
    runForrest = RandomForestClassifier(n_estimators=200, n_jobs=1, max_depth=10, max_features=5, min_samples_leaf=10, min_samples_split=20)

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