import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # import the raw data
    raw_pitch_df = pd.read_csv('./data/organized_data_for_factor.csv')
    raw_pitch_df.pitch_type.value_counts()

    # Encode the result
    encoder = LabelEncoder()
    raw_pitch_df['pitch_type'] = encoder.fit_transform(raw_pitch_df['pitch_type'])
    mapping = dict(zip(encoder.classes_, range(1, len(encoder.classes_)+1)))
    print('Mapping of Pitch_type " ', mapping)

    # some feature creation/refinement
    raw_pitch_df['is_out'] = np.where(raw_pitch_df.event.isin([
        'Bunt Groundout', 'Bunt Lineout', 'Bunt Pop Out', 'Double Play',
        'Fielders Choice Out', 'Flyout', 'Forceout', 'Grounded Into DP',
        'Groundout', 'Lineout', 'Pop Out', 'Sac Fly DP', 'Sacrifice Bunt DP',
        'Strikeout', 'Strikeout - DP', 'Triple Play']), 1, 0)
    raw_pitch_df['is_not_out'] = 1 - raw_pitch_df['is_out']
    print(raw_pitch_df)
    test_df, train_df = train_test_split(raw_pitch_df, test_size=0.1)

    # selecting features for training
    print("selecting features")
    features = ['inning',
                'top',
                'start_speed',
                'end_speed',
                'spin_rate',
                'b_count',
                's_count',
                'outs',
                'pitch_num',
                'pitch_type',
                'on_1b',
                'on_2b',
                'on_3b']
    response = 'is_out'

    # create and train random forest
    print("create and train random forest")
    runForrest = RandomForestClassifier(n_estimators=100, n_jobs=1, max_depth=10, max_features=13, min_samples_leaf=10, min_samples_split=20)
    runForrest.fit(train_df[features], train_df[response])

    probas = runForrest.predict_proba(test_df[features])
    preds = runForrest.predict(test_df[features])
    importances = runForrest.feature_importances_
    print(importances)

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
    plt.xlabel('Predicted probability of out')
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
