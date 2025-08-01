import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf

from main import categorize_age


df = pd.read_csv('df_trip.csv')
np.random.seed(6)
tf.random.set_seed(6)
def remove_corrupt_rows(df):
	df = df[~df.isin([-1, -9, -7, -8]).any(axis=1)]
	return df

def filter_loc(df):
	df = df[df.CDIVMSAR.isin([51,52,53,54])]
	return df

def create_df():
	# a = d.groupby(by="id",as_index=False).agg({"SEQ_TRIPID":min, "SEQ_TRIPID":max, "TRIPID":min,"TRIPID":max})
	# d.groupby(by=["id","TRAVDAY"]).count()

	# df_trip['date_'] = df_trip[['TDAYDATE', 'TRAVDAY']].apply(lambda x:dt.datetime(int(str(x.TDAYDATE)[:4]),int(str(x.TDAYDATE)[4:],10),x.TRAVDAY).date(), axis=1)
	# df_trip['year_'] = df_trip['TDAYDATE'].apply(lambda x:x//100)
	# df_trip['month_'] = df_trip['TDAYDATE'].apply(lambda x:x%100)
	# df_trip['day_'] = df_trip['TRAVDAY']
	# df_trip['hour_'] = df_trip['STRTTIME'].apply(lambda x:x//100)
	# df_trip['minute_'] = df_trip['STRTTIME'].apply(lambda x:x%100)
	df_per = pd.read_csv("/home/samim/pycharm_projects/academic/lstm/data/2022/perv2pub.csv", header=0,
						 usecols=['HOUSEID','PERSONID','R_SEX','WORKER','R_AGE','R_RELAT'])
	df_trip = pd.read_csv("/home/samim/pycharm_projects/academic/lstm/data/2022/tripv2pub.csv", header=0,
						  usecols=['HOUSEID','PERSONID',"SEQ_TRIPID","STRTTIME","TDAYDATE","TRAVDAY","TRPTRANS","CDIVMSAR","HHFAMINC","WHYTRP1S"])

	df_trip = filter_loc(df_trip)
	df_trip = pd.merge(df_trip, df_per, on=['HOUSEID', 'PERSONID'])
	df = remove_corrupt_rows(df_trip)
	df_trip = df_trip[df_trip.SEQ_TRIPID>1]

	df_trip = categorize_age(df_trip)
	df_trip['time_of_day'] = df_trip['STRTTIME'].apply(lambda x: ((x // 100) * 60 + (x % 100)) // 30 + 1)

	df_trip['WHYTRP1S'] = df_trip.WHYTRP1S.replace({10: 2, 40: 3, 20: 4, 30: 5, 50: 5, 70: 5, 80: 5, 97: 5})
	df_trip["id"] = df_trip[['HOUSEID', 'PERSONID']].apply(lambda x: int(f"{x.HOUSEID}{x.PERSONID}"), axis=1)
	df_trip = df_trip[
		["id", "R_RELAT", "HHFAMINC", "age_category", "R_SEX", "WORKER", "TRPTRANS", "TRAVDAY", "time_of_day", "SEQ_TRIPID", "WHYTRP1S"]]
	df_trip.drop_duplicates(inplace=True)
	dd = {1:20,2:18,3:1,4:3,5:2,6:4,7:21,8:7,9:21,10:9,11:8,12:17,13:21,14:21,15:13,16:11,17:15,18:21,19:14,20:21,97:21}
	df_trip['TRPTRANS'] = df_trip.TRPTRANS.replace({10: 11})
	df_trip = df_trip[df_trip.TRPTRANS.isin(dd.values())]
	df = df_trip.sort_values(['id', 'time_of_day'])
	return df


df = create_df()


# Preprocessing
def preprocess_data(df):
	features = ["R_RELAT", "HHFAMINC", "age_category", "R_SEX", "WORKER", "TRPTRANS", "TRAVDAY", "time_of_day", "WHYTRP1S"]
	target = 'WHYTRP1S'

	# One-hot encode categorical features
	ohe = OneHotEncoder(sparse_output=False)
	categorical_features = ['R_RELAT', "age_category", 'R_SEX', 'WORKER', 'TRPTRANS', "TRAVDAY", "time_of_day", "WHYTRP1S"]
	encoded_features = ohe.fit_transform(df[categorical_features])
	encoded_feature_names = ohe.get_feature_names_out(["R_RELAT", "age_category", 'R_SEX', 'WORKER', 'TRPTRANS', "TRAVDAY", "time_of_day", "WHYTRP1S"])
	one_hot_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)
	df = pd.concat([df, one_hot_df], axis=1)

	grouped = df.groupby('id')
	# grouped_seq = df.groupby('id').tail(7).apply(lambda g: g.iloc[:-1])
	# grouped_last = df.groupby('id').tail(1)
	X_sequences = []
	y_sequences = []
	for _, group in grouped:
		# Drop unused columns and target from input features
		X = group.tail(7).apply(lambda g: g.iloc[:-1]).drop(columns=['id', 'WHYTRP1S', "R_RELAT", "age_category", 'R_SEX', 'WORKER', 'TRPTRANS', "TRAVDAY", "time_of_day"]).values
		y = group.tail(1)[['WHYTRP1S_1', 'WHYTRP1S_2', 'WHYTRP1S_3', 'WHYTRP1S_4', 'WHYTRP1S_5']].values
		X_sequences.append(X)
		y_sequences.append(y)
	X_sequences = pad_sequences(X_sequences, maxlen=6, padding='pre', value=0.0)
	return X_sequences, y_sequences

X, y = preprocess_data(df)
# 1) Split off the test set (20% of the data)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
y_test = np.array(y_test).reshape(-1, 5)
# 2) Split the remaining 80% into train (60% overall) and val (20% overall)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)
y_train, y_val = np.array(y_train).reshape(-1, 5), np.array(y_val).reshape(-1, 5)


def create_model(trial, input_shape):
	# Hyperparameters to tune
	lstm_units = trial.suggest_int('lstm_units_1', 32, 256, step=32)
	lstm_units = trial.suggest_int('lstm_units_2', 32, 256, step=32)
	dense_units_1 = trial.suggest_int('dense_units_1', 32, 128, step=32)
	dense_units_2 = trial.suggest_int('dense_units_2', 16, 64, step=16)
	dropout_lstm = trial.suggest_float('dropout_lstm', 0.0, 0.5, step=0.1)
	dropout_dense_1 = trial.suggest_float('dropout_dense_1', 0.0, 0.5, step=0.1)
	dropout_dense_2 = trial.suggest_float('dropout_dense_2', 0.0, 0.5, step=0.1)
	learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

	model = Sequential()
	model.add(LSTM(lstm_units, input_shape=input_shape, return_sequences=False))
	model.add(Dropout(dropout_lstm))
	model.add(BatchNormalization())
	model.add(Dense(dense_units_1, activation='relu'))
	model.add(Dropout(dropout_dense_1))
	model.add(Dense(dense_units_2, activation='relu'))
	model.add(Dropout(dropout_dense_2))
	model.add(Dense(y_train.shape[1], activation='softmax'))

	optimizer = Adam(learning_rate=learning_rate)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
	return model


def objective(trial):
	# Suggest the number of epochs and batch size
	epochs = trial.suggest_int('epochs', 1, 20, step=1)
	batch_size = trial.suggest_int('batch_size', 32, 256, step=32)

	model = create_model(trial, (X_train.shape[1], X_train.shape[2]))

	# Optional early stopping for faster convergence
	es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

	# Pruning callback (optional)
	pruning_callback = TFKerasPruningCallback(trial, 'val_accuracy')

	history = model.fit(
		X_train, y_train,
		validation_data=(X_val, y_val),
		epochs=epochs,
		batch_size=batch_size,
		callbacks=[es, pruning_callback],
		verbose=0
	)

	# Get the best validation accuracy of this trial
	val_acc = max(history.history['val_accuracy'])
	return val_acc


# Create a study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10, timeout=3600)  # for example, 20 trials

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
best_trial = study.best_trial
print("  Value: ", best_trial.value)
print("  Params: ")
for key, value in best_trial.params.items():
	print("    {}: {}".format(key, value))

# Rebuild and evaluate the best model
best_model = create_model(study.best_trial, (X_train.shape[1], X_train.shape[2]))
best_model.fit(X_train, y_train, epochs=best_trial.params['epochs'], batch_size=best_trial.params['batch_size'],
			   verbose=0)
_, test_acc = best_model.evaluate(X_val, y_val, verbose=0)
print(f"Test Accuracy with best hyperparameters: {test_acc * 100:.2f}%")

