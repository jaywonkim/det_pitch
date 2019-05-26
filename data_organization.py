import pandas as pd
import csv
import warnings
warnings.filterwarnings('ignore')

def org_data(raw_pitch_df, raw_bat_df):
    to_list = raw_bat_df['ab_id'].values.tolist()
    base_data = pd.DataFrame()

    for i in range(len(raw_bat_df)):
        arranged_data = raw_pitch_df[raw_pitch_df.ab_id.isin([to_list[i]])]
        arranged_data = arranged_data.loc[:,
                        ["px", "pz", "ab_id", "pitch_type", "b_count", "s_count", "outs", "pitch_num", "on_1b", "on_2b",
                         "on_3b"]]
        arranged_data['event'] = str("N/A")
        index = arranged_data.groupby(['ab_id'])['pitch_num'].transform(max) == arranged_data['pitch_num']
        arranged_data = arranged_data.reset_index(drop=True)
        arranged_data.set_value(len(index) - 1, 'event', raw_bat_df.event[i])
        base_data = base_data.append(arranged_data)
    base_data = base_data.reset_index(drop=True)
    base_data = base_data.dropna(axis=0)
    return base_data

if __name__ == "__main__":
    # import the raw data
    raw_pitch_df = pd.read_csv('./data/DivFile/pitches000000.csv')
    raw_bat_df = pd.read_csv('./data/pujols_data.csv')

    final_data = pd.DataFrame()
    final_data = final_data.append(org_data(raw_pitch_df, raw_bat_df))

    raw_pitch_df = pd.read_csv('./data/DivFile/pitches000001.csv')
    final_data = final_data.append(org_data(raw_pitch_df, raw_bat_df))

    raw_pitch_df = pd.read_csv('./data/DivFile/pitches000002.csv')
    final_data = final_data.append(org_data(raw_pitch_df, raw_bat_df))

    final_data = final_data[final_data.event.isin([
        'Bunt Groundout', 'Bunt Lineout', 'Bunt Pop Out', 'Double Play',
        'Fielders Choice Out', 'Flyout', 'Forceout', 'Grounded Into DP',
        'Groundout', 'Lineout', 'Pop Out', 'Sac Fly DP', 'Sacrifice Bunt DP',
        'Strikeout', 'Strikeout - DP', 'Triple Play'])]
    print(final_data)
    final_data.to_csv('./data/organized_data.csv', header=True, index=False)