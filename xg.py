import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Helper functions for data cleaning
def remove_corrupt_rows(df):
    return df[~df.isin([-1, -9, -7, -8]).any(axis=1)]

def filter_loc(df):
    return df[df.CDIVMSAR.isin([51, 52, 53, 54])]

def create_df():
    # Load personal and trip data (update paths as needed)
    df_per = pd.read_csv("/home/samim/pycharm_projects/academic/lstm/data/2022/perv2pub.csv",
                         usecols=['HOUSEID', 'PERSONID', 'R_SEX', 'WORKER', 'R_AGE', 'R_RELAT'])
    df_trip = pd.read_csv("/home/samim/pycharm_projects/academic/lstm/data/2022/tripv2pub.csv",
                          usecols=['HOUSEID', 'PERSONID', "SEQ_TRIPID", "STRTTIME", "TDAYDATE",
                                   "TRAVDAY", "TRPTRANS", "CDIVMSAR", "HHFAMINC", "WHYTRP1S"])

    # Filter and merge datasets
    df_trip = filter_loc(df_trip)
    df_trip = pd.merge(df_trip, df_per, on=['HOUSEID', 'PERSONID'])
    df_trip = remove_corrupt_rows(df_trip)

    # Create a unique identifier and filter groups with more than one row
    df_trip["id"] = df_trip[['HOUSEID', 'PERSONID']].apply(lambda x: int(f"{x.HOUSEID}{x.PERSONID}"), axis=1)
    df_trip = df_trip.groupby('id').filter(lambda x: len(x) > 1)

    # Map trip purpose labels to new numerical values (example mapping)
    df_trip['WHYTRP1S'] = df_trip.WHYTRP1S.replace({10: 2, 40: 3, 20: 4, 30: 5, 50: 5, 70: 5, 80: 5, 97: 5})

    # Select only numerical features and the target variable
    # Here we use: R_RELAT, HHFAMINC, R_AGE, R_SEX, WORKER, TRPTRANS, TRAVDAY, STRTTIME
    df_trip = df_trip[['R_RELAT', 'HHFAMINC', 'R_AGE', 'R_SEX', 'WORKER',
                       'TRPTRANS', 'TRAVDAY', 'STRTTIME', 'WHYTRP1S']]
    df_trip.drop_duplicates(inplace=True)
    return df_trip

# Create the dataframe
df = create_df()

# Separate features and target
X = df.drop('WHYTRP1S', axis=1)
y = df['WHYTRP1S']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

# Initialize and train the XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=6)
model.fit(X_train, y_train)

'''
fruit_counts = df_trip['id'].value_counts()
df = pd.DataFrame({'data_values':fruit_counts.values})
data = df['data_values']
sns.distplot(data,bins="doane",kde=False,hist_kws={"align" : "left"})
plt.show()
'''


# Make predictions on the test set and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
