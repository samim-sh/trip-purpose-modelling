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

df = pd.read_csv('df_trip.csv')


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

	# Encode the target
	y_ohe = OneHotEncoder(sparse_output=False)
	y = y_ohe.fit_transform(df[target].values.reshape(-1, 1))

	X = df.drop(columns=['id', 'R_SEX', 'WORKER', 'TRPTRANS', 'R_RELAT', "TRAVDAY", 'WHYTRP1S']).values

	return X, y


X, y = preprocess_data(df)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input data for LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))


def create_model(trial, input_shape):
	# Hyperparameters to tune
	lstm_units = trial.suggest_int('lstm_units', 32, 256, step=32)
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

