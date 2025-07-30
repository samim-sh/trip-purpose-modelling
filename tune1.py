import optuna
from optuna.integration import TFKerasPruningCallback

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Replace this import with your own function that categorizes age
from main import categorize_age

np.random.seed(6)
tf.random.set_seed(6)


def remove_corrupt_rows(df):
	df = df[~df.isin([-1, -9, -7, -8]).any(axis=1)]
	return df


def filter_loc(df):
	df = df[df.CDIVMSAR.isin([51, 52, 53, 54])]
	return df


def create_df():
	df_per = pd.read_csv(
		"/home/samim/pycharm_projects/academic/lstm/data/2022/perv2pub.csv",
		header=0,
		usecols=['HOUSEID', 'PERSONID', 'R_SEX', 'WORKER', 'R_AGE', 'R_RELAT']
	)
	df_trip = pd.read_csv(
		"/home/samim/pycharm_projects/academic/lstm/data/2022/tripv2pub.csv",
		header=0,
		usecols=['HOUSEID', 'PERSONID', "SEQ_TRIPID", "STRTTIME", "TDAYDATE", "TRAVDAY",
				 "TRPTRANS", "CDIVMSAR", "HHFAMINC", "WHYTRP1S"]
	)
	df_trip = filter_loc(df_trip)
	df_trip = pd.merge(df_trip, df_per, on=['HOUSEID', 'PERSONID'])
	df_trip = remove_corrupt_rows(df_trip)

	# Create a unique ID
	df_trip["id"] = df_trip[['HOUSEID', 'PERSONID']].apply(lambda x: int(f"{x.HOUSEID}{x.PERSONID}"), axis=1)

	# Categorize age, etc.
	df_trip = categorize_age(df_trip)

	# Time of day feature
	df_trip['time_of_day'] = df_trip['STRTTIME'].apply(
		lambda x: ((x // 100) * 60 + (x % 100)) // 30 + 1
	)

	# Simplify target categories
	df_trip['WHYTRP1S'] = df_trip.WHYTRP1S.replace({
		10: 2, 40: 3, 20: 4, 30: 5,
		50: 5, 70: 5, 80: 5, 97: 5
	})

	df_trip = df_trip[
		["id", "R_RELAT", "HHFAMINC", "age_category", "R_SEX",
		 "WORKER", "TRPTRANS", "TRAVDAY", "time_of_day",
		 "SEQ_TRIPID", "WHYTRP1S"]
	]
	df_trip.drop_duplicates(inplace=True)

	# Example transformation on TRPTRANS
	dd = {
		1: 20, 2: 18, 3: 1, 4: 3, 5: 2, 6: 4, 7: 21, 8: 7, 9: 21, 10: 9,
		11: 8, 12: 17, 13: 21, 14: 21, 15: 13, 16: 11, 17: 15,
		18: 21, 19: 14, 20: 21, 97: 21
	}
	df_trip['TRPTRANS'] = df_trip.TRPTRANS.replace({10: 11})
	df_trip = df_trip[df_trip.TRPTRANS.isin(dd.values())]

	# Sort by id, then time_of_day
	df = df_trip.sort_values(['id', 'time_of_day'])
	return df


# We'll load the full dataframe once (outside the objective).
df_full = create_df()


def preprocess_data_with_window_size(df, window_size):
	"""
	Convert df into sequences of length (window_size - 1) for X
	and use the last row’s target for y.
	"""
	# One-hot encode relevant columns
	categorical_features = [
		'R_RELAT', "age_category", 'R_SEX', 'WORKER',
		'TRPTRANS', "TRAVDAY", "time_of_day", "WHYTRP1S"
	]
	ohe = OneHotEncoder(sparse_output=False)
	encoded_features = ohe.fit_transform(df[categorical_features])
	encoded_feature_names = ohe.get_feature_names_out(categorical_features)

	one_hot_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)
	df_encoded = pd.concat([df, one_hot_df], axis=1)

	# Filter out groups that have fewer rows than window_size
	df_encoded = df_encoded.groupby('id').filter(lambda g: len(g) >= window_size)

	X_sequences = []
	y_sequences = []

	grouped = df_encoded.groupby('id')
	for _, group in grouped:
		# Take the last 'window_size' rows
		sub_group = group.tail(window_size)

		# Drop columns we don't want in the model inputs
		X = sub_group.iloc[:-1].drop(
			columns=[
				'id', 'WHYTRP1S', 'R_RELAT', 'age_category', 'R_SEX',
				'WORKER', 'TRPTRANS', 'TRAVDAY', 'time_of_day'
			],
			errors='ignore'
		).values  # shape => (window_size - 1, n_features)

		# The last row’s target (one-hot)
		y = sub_group.iloc[-1][
			[c for c in encoded_feature_names if c.startswith('WHYTRP1S_')]
		].values

		# Only append if we actually got (window_size - 1) rows for X
		if len(X) == window_size - 1:
			X_sequences.append(X)
			y_sequences.append(y)

	# Pad sequences if needed (here might be unnecessary if always exact length)
	X_sequences = pad_sequences(
		X_sequences,
		maxlen=window_size - 1,
		padding='pre',
		value=0.0,
		dtype='float32'
	)

	return np.array(X_sequences), np.array(y_sequences), encoded_feature_names


# ------------------- OPTUNA SETUP ------------------- #

def create_model(trial, input_dim, num_classes):
	"""
	Build the LSTM model, sampling hyperparameters from the trial object.
	"""
	# Hyperparameters to tune
	lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 256, step=32)
	lstm_units_2 = trial.suggest_int('lstm_units_2', 32, 256, step=32)

	dropout_1 = trial.suggest_float('dropout_1', 0.0, 0.5, step=0.1)
	dropout_2 = trial.suggest_float('dropout_2', 0.0, 0.5, step=0.1)
	dropout_3 = trial.suggest_float('dropout_3', 0.0, 0.5, step=0.1)

	dense_units_1 = trial.suggest_int('dense_units_1', 32, 128, step=32)
	dense_units_2 = trial.suggest_int('dense_units_2', 16, 64, step=16)

	learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

	model = Sequential()
	# Masking to handle padded data
	model.add(Masking(mask_value=0.0, input_shape=(None, input_dim)))

	# First LSTM
	model.add(LSTM(lstm_units_1, return_sequences=True))
	model.add(Dropout(dropout_1))

	# Second LSTM
	model.add(LSTM(lstm_units_2, return_sequences=False))
	model.add(Dropout(dropout_2))
	model.add(BatchNormalization())

	# Dense layers
	model.add(Dense(dense_units_1, activation='relu'))
	model.add(Dropout(dropout_3))
	model.add(Dense(dense_units_2, activation='relu'))

	# Output layer
	model.add(Dense(num_classes, activation='softmax'))

	optimizer = Adam(learning_rate=learning_rate)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
	return model


def objective(trial):
	"""
	Objective function for Optuna:
	1) sample a window size
	2) generate sequences with that window size
	3) build and train the model
	4) return best validation accuracy
	"""
	# --------------- 1) Sample the window size  ---------------
	window_size = trial.suggest_int("window_size", 3, 10, step=1)

	# --------------- 2) Generate sequences  ---------------
	X, y, encoded_feature_names = preprocess_data_with_window_size(df_full, window_size)

	if len(X) == 0:
		# If no sequences generated at this window size, return a poor metric
		return 0.0

	# Train-test split
	X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

	num_features = X_train.shape[2]
	num_classes = y.shape[1]

	# --------------- 3) Model creation & training ---------------
	model = create_model(trial, input_dim=num_features, num_classes=num_classes)

	# Suggest batch size and epochs
	batch_size = trial.suggest_int('batch_size', 32, 512, step=32)
	epochs = trial.suggest_int('epochs', 5, 50, step=5)

	# Early stopping & pruning
	es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
	pruning_callback = TFKerasPruningCallback(trial, 'val_accuracy')

	history = model.fit(
		X_train, y_train,
		validation_data=(X_val, y_val),
		epochs=epochs,
		batch_size=batch_size,
		verbose=0,
		callbacks=[es, pruning_callback]
	)

	# 4) Return best validation accuracy
	val_acc = max(history.history['val_accuracy'])
	return val_acc


# ------------------- RUN THE STUDY ------------------- #
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, timeout=3600)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
best_trial = study.best_trial
print("  Value (Validation Accuracy): ", best_trial.value)
print("  Params: ")
for key, value in best_trial.params.items():
	print(f"    {key}: {value}")

# -------------- RETRAIN & EVALUATE WITH BEST PARAMS -------------- #
best_window_size = best_trial.params['window_size']
X_best, y_best, encoded_feature_names = preprocess_data_with_window_size(df_full, best_window_size)

# Normal train/test split
X_temp, X_test, y_temp, y_test = train_test_split(X_best, y_best, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

num_features = X_train.shape[2]
num_classes = y_best.shape[1]

best_model = create_model(best_trial, input_dim=num_features, num_classes=num_classes)

best_model.fit(
	X_train, y_train,
	validation_data=(X_val, y_val),
	epochs=best_trial.params['epochs'],
	batch_size=best_trial.params['batch_size'],
	verbose=1
)

test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy with Best Hyperparameters (including window_size): {test_acc * 100:.2f}%")

train_loss, train_acc = best_model.evaluate(X_train, y_train, verbose=0)
print(f"Train Accuracy with Best Hyperparameters: {train_acc * 100:.2f}%")
