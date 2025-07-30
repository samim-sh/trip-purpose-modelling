import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from main import categorize_age

np.random.seed(6)
tf.random.set_seed(6)
window_size = 5

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
		# Drop unused columns and target from inpu9t features
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


class TransformerDataset(Dataset):
	"""
	A simple PyTorch dataset to handle (X, y) pairs of sequences.
	"""

	def __init__(self, X, y):
		"""
		X: numpy array of shape (num_samples, seq_length, feature_dim)
		y: numpy array of shape (num_samples, num_classes)
		"""
		self.X = torch.tensor(X, dtype=torch.float32)
		self.y = torch.tensor(y, dtype=torch.float32)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]


# Create dataset objects
train_dataset = TransformerDataset(X_train, y_train)
val_dataset = TransformerDataset(X_val, y_val)
test_dataset = TransformerDataset(X_test, y_test)

# Create DataLoader objects
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ------------------------------------------------------------------
# 3) Positional Encoding Module
# ------------------------------------------------------------------

class PositionalEncoding(nn.Module):
	"""
	Implements the sinusoidal positional encoding function.
	Reference: https://arxiv.org/abs/1706.03762
	"""

	def __init__(self, d_model, max_len=5000):
		super(PositionalEncoding, self).__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		# pe shape: [max_len, d_model]
		pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
		self.register_buffer('pe', pe)

	def forward(self, x):
		"""
		x shape: [batch_size, seq_len, d_model]
		"""
		seq_len = x.size(1)
		# Add positional encoding to the input
		x = x + self.pe[:, :seq_len, :]
		return x


# ------------------------------------------------------------------
# 4) Transformer Model Definition
# ------------------------------------------------------------------

class ActivityTransformer(nn.Module):
	def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1, num_classes=5):
		"""
		:param input_dim: dimensionality of each time step's feature vector
		:param d_model: embedding dimension for the Transformer
		:param nhead: number of attention heads
		:param num_layers: how many TransformerEncoder layers to stack
		:param dim_feedforward: size of the feedforward sublayer
		:param dropout: dropout rate
		:param num_classes: number of output classes for classification
		"""
		super(ActivityTransformer, self).__init__()

		# Step 1: Project input features into d_model dimension
		self.input_projection = nn.Linear(input_dim, d_model)

		# Step 2: Positional Encoding
		self.pos_encoder = PositionalEncoding(d_model)

		# Step 3: Transformer Encoder
		encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
												   nhead=nhead,
												   dim_feedforward=dim_feedforward,
												   dropout=dropout,
												   batch_first=True)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

		# Step 4: Final classification head
		self.fc_out = nn.Sequential(
			nn.Linear(d_model, d_model // 2),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(d_model // 2, num_classes)
		)

	def forward(self, x):
		# x shape: [batch_size, seq_len, input_dim]

		# 1) Project to d_model
		x = self.input_projection(x)  # shape: [batch_size, seq_len, d_model]

		# 2) Apply positional encoding
		x = self.pos_encoder(x)  # shape: [batch_size, seq_len, d_model]

		# 3) Transformer Encoder
		#    Output shape: [batch_size, seq_len, d_model]
		x = self.transformer_encoder(x)

		# A typical approach: Take the hidden state at the last time step or do a pool
		# For classification, we can just take the mean across seq_len or the last token
		# Let's do a mean pool here:
		x = x.mean(dim=1)  # shape: [batch_size, d_model]

		# 4) Final classification head
		logits = self.fc_out(x)  # shape: [batch_size, num_classes]
		return logits


# Instantiate the model
input_dim = X_train.shape[2]  # same as feature_dim from your data
model = ActivityTransformer(input_dim=input_dim,
							d_model=128,
							nhead=4,
							num_layers=2,
							dim_feedforward=256,
							dropout=0.1,
							num_classes=num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------------------------------------------------------
# 5) Training Loop
# ------------------------------------------------------------------

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train_one_epoch(model, dataloader, optimizer, criterion):
	model.train()
	total_loss = 0
	total_correct = 0
	total_samples = 0

	for batch_X, batch_y in dataloader:
		batch_X = batch_X.to(device)  # shape: [batch_size, seq_len, feature_dim]
		batch_y = batch_y.to(device)  # shape: [batch_size, num_classes]

		optimizer.zero_grad()

		# Forward pass
		logits = model(batch_X)  # shape: [batch_size, num_classes]

		# Compute loss
		# batch_y is one-hot => convert to class indices
		targets = torch.argmax(batch_y, dim=1)  # shape: [batch_size]
		loss = criterion(logits, targets)

		# Backprop
		loss.backward()
		optimizer.step()

		total_loss += loss.item() * batch_X.size(0)

		# Calculate accuracy
		preds = torch.argmax(logits, dim=1)
		correct = (preds == targets).sum().item()
		total_correct += correct
		total_samples += batch_X.size(0)

	avg_loss = total_loss / total_samples
	accuracy = total_correct / total_samples
	return avg_loss, accuracy


def evaluate(model, dataloader, criterion):
	model.eval()
	total_loss = 0
	total_correct = 0
	total_samples = 0

	with torch.no_grad():
		for batch_X, batch_y in dataloader:
			batch_X = batch_X.to(device)
			batch_y = batch_y.to(device)

			logits = model(batch_X)
			targets = torch.argmax(batch_y, dim=1)
			loss = criterion(logits, targets)

			total_loss += loss.item() * batch_X.size(0)

			preds = torch.argmax(logits, dim=1)
			correct = (preds == targets).sum().item()
			total_correct += correct
			total_samples += batch_X.size(0)

	avg_loss = total_loss / total_samples
	accuracy = total_correct / total_samples
	return avg_loss, accuracy


# Training loop
num_epochs = 5
for epoch in range(num_epochs):
	train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
	val_loss, val_acc = evaluate(model, val_loader, criterion)

	print(f"Epoch [{epoch + 1}/{num_epochs}] "
		  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
		  f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# ------------------------------------------------------------------
# 6) Final Evaluation on Test Data
# ------------------------------------------------------------------

test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

train_loss, train_acc = evaluate(model, train_loader, criterion)
print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
