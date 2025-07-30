import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder



from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from main import categorize_age

np.random.seed(6)
tf.random.set_seed(6)
window_size = 2

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
	df_trip = remove_corrupt_rows(df_trip)

	df_trip["id"] = df_trip[['HOUSEID', 'PERSONID']].apply(lambda x: int(f"{x.HOUSEID}{x.PERSONID}"), axis=1)
	df_trip = df_trip.groupby('id').filter(lambda x: len(x) > 1)

	df_trip = categorize_age(df_trip)
	df_trip['time_of_day'] = df_trip['STRTTIME'].apply(lambda x: ((x // 100) * 60 + (x % 100)) // 30 + 1)

	df_trip['WHYTRP1S'] = df_trip.WHYTRP1S.replace({10: 2, 40: 3, 20: 4, 30: 5, 50: 5, 70: 5, 80: 5, 97: 5})
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
		X = group.tail(window_size).apply(lambda g: g.iloc[:-1]).drop(columns=['id', 'WHYTRP1S', "R_RELAT", "age_category", 'R_SEX', 'WORKER', 'TRPTRANS', "TRAVDAY", "time_of_day"]).values
		y = group.tail(1)[['WHYTRP1S_1', 'WHYTRP1S_2', 'WHYTRP1S_3', 'WHYTRP1S_4', 'WHYTRP1S_5']].values
		X_sequences.append(X)
		y_sequences.append(y)
	X_sequences = pad_sequences(X_sequences, maxlen=window_size-1, padding='pre', value=0.0)
	return X_sequences, y_sequences

X, y = preprocess_data(df)
# 1) Split off the test set (20% of the data)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
y_test = np.array(y_test).reshape(-1, 5)
# 2) Split the remaining 80% into train (60% overall) and val (20% overall)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)
y_train, y_val = np.array(y_train).reshape(-1, 5), np.array(y_val).reshape(-1, 5)
# # Reshape input data for LSTM (samples, timesteps, features)
# X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
# X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
# X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

# Build LSTM Model
num_classes = len(np.unique(df['WHYTRP1S']))

# Transformer Model
def create_transformer_model(input_dim, num_classes, embed_dim=32, num_heads=4, ff_dim=64, dropout=0.1, num_transformer_blocks=2):
	inputs = keras.Input(shape=(input_dim))

	# Embedding Layer (Crucial for non-textual data)
	x = layers.Dense(embed_dim, activation='relu')(inputs) # Map input features to higher dim space

	for _ in range(num_transformer_blocks):
		# Multi-Head Attention
		attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
		attention_output = layers.Dropout(dropout)(attention_output)
		x = layers.Add()([x, attention_output])
		x = layers.LayerNormalization(epsilon=1e-6)(x)

		# Feed Forward Part
		ff_output = layers.Dense(ff_dim, activation="relu")(x)
		ff_output = layers.Dropout(dropout)(ff_output)
		ff_output = layers.Dense(embed_dim)(ff_output)
		ff_output = layers.Dropout(dropout)(ff_output)
		x = layers.Add()([x, ff_output])
		x = layers.LayerNormalization(epsilon=1e-6)(x)

	# Classification Head
	x = layers.Flatten()(x)  # Flatten the output
	x = layers.Dense(128, activation='relu')(x)
	x = layers.Dropout(0.2)(x)
	outputs = layers.Dense(num_classes, activation='softmax')(x)

	model = keras.Model(inputs=inputs, outputs=outputs)
	return model

input_shape = (X_train.shape[1], X_train.shape[2])  # (6, 93)
num_classes = y_train.shape[1]

model = create_transformer_model(input_shape, num_classes)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=256, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")