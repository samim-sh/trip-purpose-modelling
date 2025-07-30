import tensorflow as tf

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from main import categorize_age


tf.random.set_seed(7)

df_per = pd.read_csv("/home/samim/pycharm_projects/academic/lstm/data/2022/perv2pub.csv", header=0,
                     usecols=['HOUSEID','PERSONID','R_SEX','WORKER','R_AGE','R_RELAT'])
df_trip = pd.read_csv("/home/samim/pycharm_projects/academic/lstm/data/2022/tripv2pub.csv", header=0,
                      usecols=['HOUSEID','PERSONID',"STRTTIME","TDAYDATE","TRAVDAY","TRPTRANS","CDIVMSAR","HHFAMINC","WHYTRP1S"])
df_trip = pd.merge(df_trip, df_per, on=['HOUSEID', 'PERSONID'])
# df_trip = df_trip[~df_trip.isin([-1, -9, -7, -8]).any(axis=1)]

# df_trip = df_trip[df_trip.CDIVMSAR.isin([51,52,53,54])]
df_trip = categorize_age(df_trip)
df_trip['time_of_day'] = df_trip['STRTTIME'].apply(lambda x: ((x // 100) * 60 + (x % 100)) // 30 + 1)

# df_trip['date_'] = df_trip[['TDAYDATE', 'TRAVDAY']].apply(lambda x:dt.datetime(int(str(x.TDAYDATE)[:4]),int(str(x.TDAYDATE)[4:],10),x.TRAVDAY).date(), axis=1)
# df_trip['year_'] = df_trip['TDAYDATE'].apply(lambda x:x//100)
# df_trip['month_'] = df_trip['TDAYDATE'].apply(lambda x:x%100)
# df_trip['day_'] = df_trip['TRAVDAY']
# df_trip['hour_'] = df_trip['STRTTIME'].apply(lambda x:x//100)
# df_trip['minute_'] = df_trip['STRTTIME'].apply(lambda x:x%100)
df_trip['WHYTRP1S'] = df_trip.WHYTRP1S.replace({10: 2, 40: 3, 20: 4, 30: 5, 50: 5, 70: 5, 80: 5, 97: 5})
df_trip["id"] = df_trip[['HOUSEID', 'PERSONID']].apply(lambda x: int(f"{x.HOUSEID}{x.PERSONID}"), axis=1)
df_trip = df_trip[
    ["id", "R_RELAT", "HHFAMINC", "age_category", "R_SEX", "WORKER", "TRPTRANS", "TRAVDAY", "time_of_day", "WHYTRP1S"]]
df_trip.drop_duplicates(inplace=True)
dd = {1:20,2:18,3:1,4:3,5:2,6:4,7:21,8:7,9:21,10:9,11:8,12:17,13:21,14:21,15:13,16:11,17:15,18:21,19:14,20:21,97:21}
df_trip['TRPTRANS'] = df_trip.TRPTRANS.replace({10: 11})
df_trip = df_trip[df_trip.TRPTRANS.isin(dd.values())]
df = df_trip.sort_values(['id', 'time_of_day'])

# df = pd.read_csv('df_trip.csv')


# Preprocessing
def preprocess_data(df):
    features = ['age_category', "TRAVDAY", 'time_of_day', 'HHFAMINC', 'R_SEX', 'WORKER', 'TRPTRANS', 'R_RELAT']
    target = 'WHYTRP1S'

    # One-hot encode categorical features
    ohe = OneHotEncoder(sparse_output=False)
    categorical_features = ['R_SEX', 'WORKER', 'TRPTRANS', 'R_RELAT', "TRAVDAY"]
    encoded_features = ohe.fit_transform(df[categorical_features])
    encoded_feature_names = ohe.get_feature_names_out(['R_SEX', 'WORKER', 'TRPTRANS', 'R_RELAT', "TRAVDAY"])
    one_hot_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)
    df = pd.concat([df, one_hot_df], axis=1)

    X = df.drop(columns=['id', 'R_SEX', 'WORKER', 'TRPTRANS', 'R_RELAT', "TRAVDAY","WHYTRP1S"]).values
    y = df['WHYTRP1S'].values
    y = ohe.fit_transform(y.reshape(-1,1))

    return X, y

X, y = preprocess_data(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input data for LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build LSTM Model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=160, verbose=1)

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
# Evaluate the model
_, accuracy = model.evaluate(X_train, y_train, verbose=0)
print(f"train Accuracy: {accuracy * 100:.2f}%")

# plot()