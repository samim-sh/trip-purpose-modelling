import tensorflow as tf

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout, BatchNormalization, TimeDistributed, Masking
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model


def hh():
	"""
	HOUSEID - Unique Identifier- Household
		WTHHFIN - 7-day national household weight
		WTHHFIN5D - 5-day national household weight
		WTHHFIN2D - 2-day national household weight
	NUMADLT - Count of adult household members at least 18 years old
	HOMEOWN - Home ownership status (e.g., owned, rented)
		HOMETYPE - Type of home (e.g., one-family detached, apartment)
	RAIL - MSA heavy rail status for household (Yes/No)
	CENSUS_D - Census division classification for home address
	CENSUS_R - Census region classification for home address
	HH_HISP - Hispanic status of household respondent
	DRVRCNT - Number of drivers in the household
		CNTTDHH - Count of household trips on travel day
	CDIVMSAR - Grouping of household by combination of Census division, MSA status, and presence of rail
		FLAG100 - Indicator if all eligible household members completed the survey
	HHFAMINC - Household income
	HHFAMINC_IMP - Household income (imputed)
	HH_RACE - Race of household respondent
	HHSIZE - Total number of people in household
	HHVEHCNT - Total number of vehicles in household
		HHRELATD - Flag indicating at least two persons in household are related
	LIF_CYC - Life cycle classification for the household
	MSACAT - MSA category for the household home address
	MSASIZE - Population size category of the MSA
	TRAVDAY - Travel day - day of the week
	URBAN - Household urban area classification, based on 2020 TIGER/Line Shapefile
	URBANSIZE - Urban area size where home address is located
	URBRUR - Household in urban/rural area
		PPT517 - Count of household members aged 5-17 years
		YOUNGCHILD - Count of household members under 5 years old
		RESP_CNT - Count of responding persons in household
		URBRUR_2010 - Household in urban/rural area based on 2010 Census
	TDAYDATE - Date of travel day (YYYYMM)
	WRKCOUNT - Count of workers in household
	STRATUMID - Household Stratum ID for sampling purposes
	"""
	df_hh = pd.read_csv('/home/samim/pycharm_projects/academic/lstm/data/hhv2pub.csv')


def per():
	"""
	HOUSEID - Unique Identifier- Household
	PERSONID - Person ID within household
	WTPERFIN - 7 day National person weight
	WTPERFIN5D - 5 day National person weight
	WTPERFIN2D - 2 day National person weight
	R_AGE - Respondent age
	R_SEX - Respondent sex
	R_RELAT - Relationship to household respondent
	WORKER - Employment status of respondent
	DRIVER - Driver status of household member
	R_RACE - Respondent race
	GCDWORK - Great circle distance from home to work
	OUTOFTWN - Out of town status
	USEPUBTR - Used public transit on trip
	R_RACE_IMP - Respondent race (imputed)
	R_HISP - Hispanic status of respondent
	PROXY - Survey completed by self or someone else
	WHOPROXY - Who served as proxy respondent
	EDUC - Education level of respondent
	LAST30_TAXI - Used taxi in last 30 days
	LAST30_RDSHR - Used rideshare in last 30 days
	LAST30_ESCT - Used e-scooter in last 30 days
	LAST30_PT - Used public transportation in last 30 days
	LAST30_MTRC - Used motorcycle in last 30 days
	LAST30_WALK - Walked in last 30 days
	LAST30_BIKE - Bicycled in last 30 days
	LAST30_BKSHR - Used bikeshare in last 30 days
	TAXISERVICE - Availability of taxi service in area
	RIDESHARE22 - Used rideshare in 2022
	ESCOOTERUSED - E-scooter use status
	PTUSED - Public transportation use status
	TRNPASS - Transit pass ownership
	MCTRANSIT - Main mode of transit
	WALKTRANSIT - Walking as transit mode
	BIKETRANSIT - Biking as transit mode
	BIKESHARE22 - Used bikeshare in 2022
	USAGE1 - Frequency of mode usage - 1
	USAGE2_1 - Frequency of mode usage - 2 (entry 1)
	USAGE2_2 - Frequency of mode usage - 2 (entry 2)
	USAGE2_3 - Frequency of mode usage - 2 (entry 3)
	USAGE2_4 - Frequency of mode usage - 2 (entry 4)
	USAGE2_5 - Frequency of mode usage - 2 (entry 5)
	USAGE2_6 - Frequency of mode usage - 2 (entry 6)
	USAGE2_7 - Frequency of mode usage - 2 (entry 7)
	USAGE2_8 - Frequency of mode usage - 2 (entry 8)
	USAGE2_9 - Frequency of mode usage - 2 (entry 9)
	USAGE2_10 - Frequency of mode usage - 2 (entry 10)
	QACSLAN1 - Quality assessment: language 1
	QACSLAN3 - Quality assessment: language 3
	PAYPROF - Payment profile
	PRMACT - Primary activity
	EMPLOYMENT2 - Employment status (detailed)
	DRIVINGOCCUPATION - Occupation related to driving
	DRIVINGVEHICLE - Driving vehicle type
	WRKLOC - Work location status
	WKFMHM22 - Work from home status in 2022
	WRKTRANS - Mode of transportation to work
	EMPPASS - Employer-provided transit pass
	SCHOOL1 - School attendance status
	STUDE - Student status
	SCHTYP - School type
	SCHOOL1C - School attendance classification
	SCHTRN1 - Mode of transportation to school
	DELIVER - Delivery services used
	DELIV_GOOD - Goods delivery status
	DELIV_FOOD - Food delivery status
	DELIV_GROC - Grocery delivery status
	DELIV_PERS - Personal items delivery status
	RET_HOME - Return home
	RET_PUF - Return to pickup point
	RET_AMZ - Return items to Amazon
	RET_STORE - Return items to store
	MEDCOND - Medical condition status
	MEDCOND6 - Medical condition in the last 6 months
	W_CANE - Use of cane
	W_WKCR - Use of walker
	W_VISIMP - Visual impairment status
	W_SCCH - Use of special chair
	W_CHAIR - Use of wheelchair
	W_NONE - No special equipment used
	CONDTRAV - Condition affecting travel
	CONDRIDE - Condition affecting ride
	CONDNIGH - Condition affecting night travel
	CONDRIVE - Condition affecting driving
	CONDPUB - Condition affecting public transit use
	CONDSPEC - Specific condition description
	CONDSHARE - Condition affecting shared ride
	CONDNONE - No condition affecting travel
	CONDRF - Condition refusal
	FRSTHM - First-time home ownership status
	PARK - Parking availability
	PARKHOME - Home parking availability
	PARKHOMEAMT - Amount paid for parking at home
	PARKHOMEAMT_PAMOUNT - Payment amount for parking at home
	PARKHOMEAMT_PAYTYPE - Payment type for parking at home
	SAMEPLC - Same place travel status
	COV1_WK - COVID-19 work changes
	COV1_SCH - COVID-19 school changes
	COV1_PT - COVID-19 public transit changes
	COV1_OHD - COVID-19 other household changes
	COV2_WK - COVID-19 work changes (2nd assessment)
	COV2_SCH - COVID-19 school changes (2nd assessment)
	COV2_PT - COVID-19 public transit changes (2nd assessment)
	COV2_OHD - COVID-19 other household changes (2nd assessment)
	CNTTDTR - Count of total trips taken
	R_SEX_IMP - Respondent sex (imputed)
	NUMADLT - Number of adults in household
	HOMEOWN - Home ownership status
	RAIL - Presence of rail in Metropolitan Statistical Area (MSA)
	CENSUS_D - Census Division
	CENSUS_R - Census Region
	HH_HISP - Hispanic status of household respondent
	DRVRCNT - Driver count in household
	CDIVMSAR - Census Division, MSA, Rail Grouping
	HHFAMINC - Household family income
	HH_RACE - Household respondent race
	HHSIZE - Household size
	HHVEHCNT - Household vehicle count
	LIF_CYC - Household life cycle
	MSACAT - Metropolitan Statistical Area category
	MSASIZE - MSA size
	TRAVDAY - Travel day of the week
	URBAN - Urban classification of household
	URBANSIZE - Urban area size classification
	URBRUR - Urban or rural classification of household
	TDAYDATE - Travel day date (YYYYMM)
	WRKCOUNT - Worker count in household
	STRATUMID - Household Stratum ID
	HHFAMINC_IMP - Household family income (imputed)
	"""


def veh():
	"""
	HOUSEID - Unique Identifier- Household​(codebook)
	VEHID - Vehicle ID within household​(codebook)
	VEHYEAR - Vehicle year​(codebook)
	MAKE - Vehicle make​(codebook)
	HHVEHCNT - Total number of vehicles in household​(codebook)
	VEHTYPE - Vehicle type​(codebook)
	VEHFUEL - Type of fuel vehicle runs on​(codebook)
	VEHCOMMERCIAL - Vehicle used for business purposes​(codebook)
	VEHCOM_RS - Vehicle used for rideshare​(codebook)
	VEHCOM_DEL - Vehicle used for delivery service​(codebook)
	VEHCOM_OTH - Vehicle used for other business purposes​(codebook)
	COMMERCIALFREQ - Frequency of business use (not available in search results)
	HHVEHUSETIME_RS - Household vehicle use time for rideshare (not available in search results)
	HHVEHUSETIME_DEL - Household vehicle use time for delivery (not available in search results)
	HHVEHUSETIME_OTH - Household vehicle use time for other purposes (not available in search results)
	VEHOWNED - Vehicle owned for 1 year or more​(codebook)
	WHOMAIN - Main driver of vehicle​(codebook)
	VEHCASEID - Unique vehicle identifier​(codebook)
	ANNMILES - Annual miles driven by vehicle (not available in search results)
	HYBRID - Hybrid vehicle status (Yes/No)​(codebook)
	VEHAGE - Age of vehicle​(codebook)
	VEHOWNMO - Number of months the vehicle has been owned​(codebook)
	NUMADLT - Count of adult household members at least 18 years old​(codebook)
	HOMEOWN - Home ownership status​(codebook)
	RAIL - MSA heavy rail status for household​(codebook)
	CENSUS_D - Census Division classification for home address (not available in search results)
	CENSUS_R - Census Region classification for home address​(codebook)
	HH_HISP - Hispanic status of household respondent​(codebook)
	DRVRCNT - Number of drivers in the household​(codebook)
	CDIVMSAR - Census Division, MSA, Rail Grouping (not available in search results)
	HHFAMINC - Household family income​(codebook)
	HH_RACE - Household respondent race​(codebook)
	HHSIZE - Total number of people in household​(codebook)
	LIF_CYC - Life cycle classification for household​(codebook)
	MSACAT - MSA category for household home address​(codebook)
	MSASIZE - Population size category of the MSA​(codebook)
	TRAVDAY - Travel day of the week​(codebook)
	URBAN - Urban classification of household​(codebook)
	URBANSIZE - Urban area size classification​(codebook)
	URBRUR - Urban or rural classification of household​(codebook)
	TDAYDATE - Date of travel day (YYYYMM)​(codebook)
	WRKCOUNT - Worker count in household​(codebook)
	STRATUMID - Household Stratum ID for sampling purposes​(codebook)
	WTHHFIN - 7-day national household weight​(codebook)
	WTHHFIN5D - 5-day national household weight​(codebook)
	WTHHFIN2D - 2-day national household weight​(codebook)
	HHFAMINC_IMP - Household family income (imputed)​(codebook)
	"""


def ld():
	"""
	HOUSEID - Unique Identifier- Household​(codebook)
	PERSONID - Person ID within household​(codebook)
	LONGDIST - Long-distance travel indicator​(codebook)
	MAINMODE - Mode of travel for the last long-distance trip​(codebook)
	INT_FLAG - Trip was in the US or outside the US​(codebook)
	LD_NUMONTRP - Number of people with respondent on long-distance trip​(codebook)
	ONTP_P1 - Person 1 on long-distance trip​(codebook)
	ONTP_P2 - Person 2 on long-distance trip​(codebook)
	ONTP_P3 - Person 3 on long-distance trip​(codebook)
	ONTP_P4 - Person 4 on long-distance trip​(codebook)
	ONTP_P5 - Person 5 on long-distance trip​(codebook)
	ONTP_P6 - Person 6 on long-distance trip​(codebook)
	ONTP_P7 - Person 7 on long-distance trip​(codebook)
	ONTP_P8 - Person 8 on long-distance trip​(codebook)
	ONTP_P9 - Person 9 on long-distance trip​(codebook)
	ONTP_P10 - Person 10 on long-distance trip​(codebook)
	FARREAS - Farthest destination was in the US or outside​(codebook)
	LD_AMT - Amount of time spent on long-distance trip (not available in search results)
	LD_ICB - Indicates if the trip is complete (not available in search results)
	LDT_FLAG - Flag for long-distance trips of 50 miles or more​(codebook)
	BEGTRIP - Start time of the trip (not available in search results)
	ENDTRIP - End time of the trip (not available in search results)
	NTSAWAY - Nights away on long-distance trip​(codebook)
	WEEKEND - Trip occurred during the weekend (not available in search results)
	MRT_DATE - Date of most recent long-distance trip (YYYYMM)​(codebook)
	FARCDIV - Census Division for farthest destination (not available in search results)
	FARCREG - Census Region for farthest destination (not available in search results)
	GCDTOT - Great circle distance from home to the farthest domestic destination​(codebook)
	AIRSIZE - Airplane size used on trip (not available in search results)
	EXITCDIV - Census Division for exit (not available in search results)
	GCD_FLAG - Flag indicating great circle distance (not available in search results)
	NUMADLT - Count of adult household members at least 18 years old​(codebook)
	HOMEOWN - Home ownership status​(codebook)
	RAIL - MSA heavy rail status for household​(codebook)
	CENSUS_D - Census Division classification for home address (not available in search results)
	CENSUS_R - Census Region classification for home address​(codebook)
	HH_HISP - Hispanic status of household respondent​(codebook)
	DRVRCNT - Number of drivers in the household​(codebook)
	CDIVMSAR - Census Division, MSA, Rail Grouping (not available in search results)
	HHFAMINC - Household family income​(codebook)
	HHFAMINC_IMP - Household family income (imputed)​(codebook)
	HH_RACE - Race of household respondent​(codebook)
	HHSIZE - Total number of people in the household​(codebook)
	HHVEHCNT - Total number of vehicles in household​(codebook)
	LIF_CYC - Life cycle classification for household​(codebook)
	MSACAT - MSA category for the household home address​(codebook)
	MSASIZE - Population size category of the MSA​(codebook)
	TRAVDAY - Travel day of the week​(codebook)
	URBAN - Urban classification of household​(codebook)
	URBANSIZE - Urban area size classification​(codebook)
	URBRUR - Urban or rural classification of household​(codebook)
	TDAYDATE - Date of travel day (YYYYMM)​(codebook)
	WRKCOUNT - Worker count in household​(codebook)
	STRATUMID - Household Stratum ID for sampling purposes​(codebook)
	WTPERFIN - National person weight (final)​(codebook)
	WTPERFIN5D - 5-day national person weight​(codebook)
	WTPERFIN2D - 2-day national person weight​(codebook)
	R_AGE - Respondent age​(codebook)
	R_SEX - Respondent sex​(codebook)
	WORKER - Employment status of respondent​(codebook)
	DRIVER - Driver status of the respondent​(codebook)
	R_RACE - Respondent race​(codebook)
	R_HISP - Hispanic status of respondent​(codebook)
	PROXY - Survey completed by self or someone else​(codebook)
	EDUC - Education level of respondent (not available in search results)
	R_SEX_IMP - Respondent sex (imputed)​(codebook)
	"""


def trip():
	"""
	HOUSEID - Unique Identifier- Household​(codebook)
	PERSONID - Person ID within household​(codebook)
	TRIPID - Trip ID for each trip a person took​(codebook)
	SEQ_TRIPID - Renumbered sequential trip ID​(codebook)
	VEHCASEID - Unique vehicle identifier​(codebook)
	FRSTHM - First trip of the day was from home​(codebook)
		PARK - Paid for parking at any time during travel day​(codebook)
		HHMEMDRV - Household member drove​(codebook)
	TDWKND - Weekend trip indicator​(codebook)
	TRAVDAY - Travel day - day of the week​(codebook)
	LOOP_TRIP - Trip origin and destination at identical location​(codebook)
	DWELTIME - Dwelling time at location​(codebook)
	PUBTRANS - Used public transit on trip​(codebook)
		TRIPPURP - General purpose of trip​(codebook)
		WHYFROM - Purpose of starting location​(codebook)
		WHYTRP1S - Travel day trip purpose​(codebook)
	TRVLCMIN - Trip duration in minutes​(codebook)
	STRTTIME - 24 hour local start time of trip​(codebook)
	ENDTIME - 24 hour local end time of trip​(codebook)
		TRPHHVEH - Household vehicle used for trip​(codebook)
		VEHID - Vehicle ID of vehicle used from household roster​(codebook)
	TRPTRANS - Trip mode, derived (e.g., car, bus, bike)​(codebook)
		NUMONTRP - Number of people on trip​(codebook)
		ONTD_P1 - Person 1 was on trip​(codebook)
		ONTD_P2 - Person 2 was on trip​(codebook)
		ONTD_P3 - Person 3 was on trip​(codebook)
		ONTD_P4 - Person 4 was on trip​(codebook)
		ONTD_P5 - Person 5 was on trip​(codebook)
		ONTD_P6 - Person 6 was on trip​(codebook)
		ONTD_P7 - Person 7 was on trip​(codebook)
		ONTD_P8 - Person 8 was on trip​(codebook)
		ONTD_P9 - Person 9 was on trip​(codebook)
		ONTD_P10 - Person 10 was on trip​(codebook)
		NONHHCNT - Number of non-household members on trip​(codebook)
		HHACCCNT - Household members accounted for on trip​(codebook)
		WHODROVE - Who drove the vehicle​(codebook)
		DRVR_FLG - Flag for driver on trip​(codebook)
		PSGR_FLG - Flag for passenger on trip​(codebook)
		WHODROVE_IMP - Imputed value for who drove the vehicle​(codebook)
		PARK2_PAMOUNT - Amount paid for parking​(codebook)
		PARK2_PAYTYPE - Periodicity of parking payment​(codebook)
		PARK2 - Paid for parking on this trip​(codebook)
	WHYTO - Purpose of ending location​(codebook)
	WALK - Walking activity on trip​(codebook)
	TRPMILES - Calculated trip distance converted into miles​(codebook)
		WTTRDFIN - Final trip weight for travel day​(codebook)
		WTTRDFIN5D - 5-day trip weight for travel day​(codebook)
		WTTRDFIN2D - 2-day trip weight for travel day​(codebook)
		TDCASEID - Unique identifier for every trip record in the file​(codebook)
		VMT_MILE - Vehicle miles traveled during trip​(codebook)
		GASPRICE - Gasoline price during trip (not available in search results)
		WHYTRP90 - Travel day trip purpose consistent with 1990 NPTS design​(codebook)
		NUMADLT - Count of adult household members at least 18 years old​(codebook)
		HOMEOWN - Home ownership status​(codebook)
	RAIL - MSA heavy rail status for household​(codebook)
	CENSUS_D - Census Division classification for home address (not available in search results)
	CENSUS_R - Census Region classification for home address​(codebook)
		HH_HISP - Hispanic status of household respondent​(codebook)
		DRVRCNT - Driver count in household​(codebook)
	CDIVMSAR - Census Division, MSA, Rail Grouping​(codebook)
	HHFAMINC - Household family income​(codebook)
		HH_RACE - Race of household respondent​(codebook)
		HHSIZE - Total number of people in the household​(codebook)
		HHVEHCNT - Total number of vehicles in the household​(codebook)
		LIF_CYC - Life cycle classification for the household​(codebook)
		MSACAT - MSA category for household home address​(codebook)
		MSASIZE - Population size category of the MSA​(codebook)
	URBAN - Household urban area classification​(codebook)
	URBANSIZE - Urban area size classification​(codebook)
	URBRUR - Household in urban/rural area​(codebook)
	TDAYDATE - Date of travel day (YYYYMM)​(codebook)
	WRKCOUNT - Count of workers in household​(codebook)
		STRATUMID - Household Stratum ID for sampling purposes​(codebook)
		R_AGE - Respondent age​(codebook)
		R_SEX - Respondent sex​(codebook)
		WORKER - Employment status of respondent​(codebook)
		DRIVER - Driver status of the respondent​(codebook)
		R_RACE - Respondent race​(codebook)
		R_HISP - Hispanic status of respondent​(codebook)
		PROXY - Survey completed by self or someone else​(codebook)
		EDUC - Education level of respondent​(codebook)
		PRMACT - Primary activity for those who did not work for pay last week​(codebook)
		R_SEX_IMP - Respondent sex (imputed)​(codebook)
		VEHTYPE - Vehicle type​(codebook)
		HHFAMINC_IMP - Household family income (imputed)​(codebook)
	"""


def neural_network(df):

	"""
		HOUSEID - Unique Identifier- Household​(codebook)
		PERSONID - Person ID within household​(codebook)
		TRIPID - Trip ID for each trip a person took​(codebook)
		SEQ_TRIPID - Renumbered sequential trip ID​(codebook)
		VEHCASEID - Unique vehicle identifier​(codebook)
		FRSTHM - First trip of the day was from home​(codebook)
		TDWKND - Weekend trip indicator​(codebook)
		TDAYDATE - Date of travel day (YYYYMM)​(codebook)
		TRAVDAY - Travel day - day of the week​(codebook)
	STRTTIME - 24 hour local start time of trip​(codebook)
		ENDTIME - 24 hour local end time of trip​(codebook)
		TRVLCMIN - Trip duration in minutes​(codebook)
		LOOP_TRIP - Trip origin and destination at identical location​(codebook)
		DWELTIME - Dwelling time at location​(codebook)
		PUBTRANS - Used public transit on trip​(codebook)
	TRPTRANS - Trip mode, derived (e.g., car, bus, bike)​(codebook)
		WALK - Walking activity on trip​(codebook)
		TRPMILES - Calculated trip distance converted into miles​(codebook)
		RAIL - MSA heavy rail status for household​(codebook)
		CENSUS_D - Census Division classification for home address (not available in search results)
		CENSUS_R - Census Region classification for home address​(codebook)
	CDIVMSAR - Census Division, MSA, Rail Grouping​(codebook)
	HHFAMINC - Household family income​(codebook)
		URBAN - Household urban area classification​(codebook)
		URBANSIZE - Urban area size classification​(codebook)
		URBRUR - Household in urban/rural area​(codebook)
		WRKCOUNT - Count of workers in household​(codebook)
	WHYTRP1S - Travel day trip purpose​(codebook)
	"""


def categorize_age(df, age_column="R_AGE"):
	bins = [-1, 10, 18, 40, 60, 80, float('inf')]
	labels = [1, 2, 3, 4, 5, 6]  # Corresponding categories for the bins
	df['age_category'] = pd.cut(df[age_column], bins=bins, labels=labels)
	return df


def plot(history):
	import matplotlib.pyplot as plt

	# Plot Accuracy
	plt.plot(history.history['accuracy'], label='Training Accuracy')
	plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
	plt.legend()
	plt.title('Training and Validation Accuracy')
	plt.show()

	# Plot Loss
	plt.plot(history.history['loss'], label='Training Loss')
	plt.plot(history.history['val_loss'], label='Validation Loss')
	plt.legend()
	plt.title('Training and Validation Loss')
	plt.show()

def tmp0(df):


	# # Load your data
	# df = pd.read_csv('df_trip.csv')  # Replace with your file path

	# Step 1: Cyclical Encoding for 'time_of_day'
	df['time_of_day_sin'] = np.sin(2 * np.pi * df['time_of_day'] / 48)
	df['time_of_day_cos'] = np.cos(2 * np.pi * df['time_of_day'] / 48)

	# Step 2: One-Hot Encoding for Low-Cardinality Features ('R_SEX', 'WORKER')
	encoder = OneHotEncoder(sparse_output=False)
	one_hot_features = encoder.fit_transform(df[["R_SEX","WORKER","R_RELAT","TRPTRANS","WHYTRP1S"]])

	# Add encoded features back to the dataframe
	encoded_feature_names = encoder.get_feature_names_out(['R_SEX','WORKER',"R_RELAT","TRPTRANS","WHYTRP1S"])
	one_hot_df = pd.DataFrame(one_hot_features, columns=encoded_feature_names, index=df.index)
	df = pd.concat([df, one_hot_df], axis=1)

	# Step 3: Prepare Sequences by Grouping by 'id'
	grouped = df.groupby('id')
	X_sequences = []
	y_sequences = []

	for _, group in grouped:
		# Drop unused columns and target from input features
		X = group.drop(columns=['id','WHYTRP1S','time_of_day','R_SEX','WORKER',"R_RELAT","TRPTRANS",
								"WHYTRP1S_1","WHYTRP1S_2","WHYTRP1S_3","WHYTRP1S_4","WHYTRP1S_5"]).values
		y = group[['WHYTRP1S_1','WHYTRP1S_2','WHYTRP1S_3','WHYTRP1S_4','WHYTRP1S_5']].values
		X_sequences.append(X)
		y_sequences.append(y)

	# # # Encode Target (y) Using Label Encoding
	# label_encoder = LabelEncoder()
	# y_encoded = [label_encoder.fit_transform(seq) for seq in y_sequences]
	#
	# # # Convert target to one-hot encoding
	num_classes = len(np.unique(df['WHYTRP1S']))
	# y_one_hot = [to_categorical(seq, num_classes=num_classes) for seq in y_encoded]
	#
	# # Take only the last label of each sequence
	# y_final_one_hot = np.array([seq[-1] for seq in y_one_hot])

	# Split Data into Train-Test Sets
	X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

	# Convert lists of variable-length sequences into TensorFlow Dataset objects
	def create_tf_dataset(X, y):
		dataset = tf.data.Dataset.from_generator(
			lambda: zip(X, y),
			output_signature=(
				tf.TensorSpec(shape=(None, X[0].shape[1]), dtype=tf.float32),
				tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),
			)
		)
		return dataset.padded_batch(32)

	train_dataset = create_tf_dataset(X_train, y_train)
	test_dataset = create_tf_dataset(X_test, y_test)

	# Build the LSTM Model
	model = Sequential()
	model.add(Masking(mask_value=0.0, input_shape=(None, X_train[0].shape[1])))  # Mask variable-length sequences
	model.add(LSTM(64, return_sequences=True))
	model.add(Dropout(0.3))
	model.add(LSTM(32, return_sequences=False))
	model.add(Dropout(0.3))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))

	# Compile the model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	# Train the model
	history = model.fit(train_dataset, validation_data=test_dataset, epochs=10)

	# Evaluate the model
	test_loss, test_accuracy = model.evaluate(test_dataset)
	print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


def tmp1(df):
	import pandas as pd
	import numpy as np
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
	from tensorflow.keras.utils import to_categorical


	# Inspect the dataset
	print("Dataset Head:")
	print(df.head())
	print("\nDataset Info:")
	print(df.info())

	# Handle missing values (if any)
	df = df.dropna()

	# Target variable
	output_column = 'WHYTRP1S'

	# Encode target variable (1 to 5 categories)
	label_encoder = LabelEncoder()
	df[output_column] = label_encoder.fit_transform(df[output_column])

	# Define feature columns
	feature_columns = ['R_SEX', 'age_category', 'WORKER', 'time_of_day', 'TRPTRANS', 'HHFAMINC', 'R_RELAT']

	# Process categorical features
	encoded_features = []
	encoders = {}

	for col in feature_columns:
		if df[col].dtype == 'object' or df[col].nunique() < 10:
			# Use OneHotEncoder or Embedding based on cardinality
			encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
			encoded = encoder.fit_transform(df[[col]])
			encoded_features.append(encoded)
			encoders[col] = encoder
		else:
			# Use OneHotEncoder or Embedding based on cardinality
			encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
			encoded = encoder.fit_transform(df[[col]])
			encoded_features.append(encoded)
			encoders[col] = encoder

	# Concatenate processed features
	X = np.hstack(encoded_features)
	y = to_categorical(df[output_column])

	# Train-Test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Define LSTM model
	model = Sequential()
	model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
	model.add(Dropout(0.2))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))

	# Compile the model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	# Train the model
	history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32, verbose=1)

	# Evaluate the model
	eval_results = model.evaluate(X_test, y_test, verbose=1)
	print(f"Test Loss: {eval_results[0]}, Test Accuracy: {eval_results[1]}")

	# Save the model
	model.save('lstm_trip_purpose_model.h5')

	# Summarize encoding and model
	print("Encoding methods:")
	for col, enc in encoders.items():
		print(f"{col}: {type(enc).__name__}")


def tmp_transformer(data):
	import pandas as pd
	import numpy as np
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import LabelEncoder
	import torch
	import torch.nn as nn
	from torch.utils.data import Dataset, DataLoader

	# # Load the dataset
	# data = pd.read_csv('/mnt/data/df_trip.csv')

	# Step 1: Preprocess the categorical data
	label_encoders = {}
	for column in data.columns:
		if data[column].dtype == 'object' or data[column].dtype.name == 'category':
			le = LabelEncoder()
			data[column] = le.fit_transform(data[column].astype(str))
			label_encoders[column] = le

	# Split features and target (Assume 'WHYTRP1S' is the target variable)
	X = data.drop('WHYTRP1S', axis=1)
	y = data['WHYTRP1S']

	# Train-test split
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

	# Convert to PyTorch tensors
	X_train = torch.tensor(X_train.values, dtype=torch.long)
	y_train = torch.tensor(y_train.values, dtype=torch.long)
	X_val = torch.tensor(X_val.values, dtype=torch.long)
	y_val = torch.tensor(y_val.values, dtype=torch.long)

	# Limit the dataset size to avoid memory issues
	max_samples = 100000  # Adjust this value based on your system's memory
	if len(X_train) > max_samples:
		X_train = X_train[:max_samples]
		y_train = y_train[:max_samples]
	if len(X_val) > max_samples:
		X_val = X_val[:max_samples]
		y_val = y_val[:max_samples]

	# Step 2: Create Dataset and DataLoader
	class TabularDataset(Dataset):
		def __init__(self, X, y):
			self.X = X
			self.y = y

		def __len__(self):
			return len(self.y)

		def __getitem__(self, idx):
			return self.X[idx], self.y[idx]

	train_dataset = TabularDataset(X_train, y_train)
	val_dataset = TabularDataset(X_val, y_val)

	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

	# Step 3: Define the Transformer model
	class TransformerModel(nn.Module):
		def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers, num_classes):
			super(TransformerModel, self).__init__()
			self.embedding = nn.Embedding(input_dim, embed_dim)
			encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
			self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
			self.fc = nn.Linear(embed_dim, num_classes)

		def forward(self, x):
			x = self.embedding(x)
			x = self.transformer(x)
			x = x.mean(dim=1)  # Global average pooling
			x = self.fc(x)
			return x

	# Correct input_dim calculation
	input_dim = int(X_train.max().max()) + 1  # Ensure input_dim covers all unique indices in the data
	embed_dim = 16
	num_heads = 2
	ff_dim = 64
	num_layers = 2
	num_classes = len(y.unique())

	model = TransformerModel(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads,
							 ff_dim=ff_dim, num_layers=num_layers, num_classes=num_classes)

	# Step 4: Training the model
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
		for epoch in range(epochs):
			model.train()
			train_loss = 0.0
			for X_batch, y_batch in train_loader:
				optimizer.zero_grad()
				outputs = model(X_batch)
				loss = criterion(outputs, y_batch)
				loss.backward()
				optimizer.step()
				train_loss += loss.item()

			val_loss = 0.0
			correct = 0
			total = 0
			model.eval()
			with torch.no_grad():
				for X_batch, y_batch in val_loader:
					outputs = model(X_batch)
					loss = criterion(outputs, y_batch)
					val_loss += loss.item()

					_, predicted = torch.max(outputs, 1)
					total += y_batch.size(0)
					correct += (predicted == y_batch).sum().item()

			accuracy = 100 * correct / total
			print(
				f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Accuracy: {accuracy}%")

	# Train the model
	train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)


if __name__ == '__main__':

	'''
	2022
	01=Spouse/Partner
	02=Child
	03=Parent
	04=Brother/Sister
	05=Other relative
	06=Not related
	07=Self
	
	2017
	01: 7=Self
	02: 1=Spouse/Unmarried partner
	03: 2=Child 
	04: 3=Parent
	05: 4=Brother/Sister
	06: 5=Other relative
	07: 6=Non-relative 
	
	2022
	01=Car 
	02=Van 
	03=SUV/Crossover 
	04=Pickup truck 
	06=Recreational 
	07=Motorcycle 
	08=Public or commuter bus
	09=School bus 
	10=Street car or trolley car 
	11=Subway or elevated rail 
	12=Commuter rail 
	13=Amtrak 
	14=Airplane 
	15=Taxicab or limo service 
	16=Other ride-sharing service 
	17=Paratransit/ Dial a ride 
	18=Bicycle (including bikeshare, ebike, etc.) 
	19=E-scooter 
	20=Walked 
	21=Other (specify)


	2017
	01 20=Walk 
	02 18=Bicycle 
	03 1=Car 
	04 3=SUV 
	05 2=Van 
	06 4=Pickup truck 
	07 21=Golf cart / Segway 
	08 7=Motorcycle / Moped 
	09 21=RV (motor home, ATV, snowmobile) 
	10 9=School bus 
	11 8=Public or commuter bus 
	12 17=Paratransit / Dial-a-ride 
	13 21=Private / Charter / Tour / Shuttle bus 
	14 21=City-to-city bus (Greyhound, Megabus) 
	15 13=Amtrak / Commuter rail 
	16 11=Subway / elevated / light rail / street car 
	17 15=Taxi / limo (including Uber / Lyft) 
	18 21=Rental car (Including Zipcar / Car2Go) 
	19 14=Airplane 
	20 21=Boat / ferry / water taxi 
	97 21=Something Else
	'''

