import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from main import categorize_age
import datetime as dt

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

    # df_trip['date_'] = df_trip[['TDAYDATE', 'TRAVDAY']].apply(lambda x:dt.datetime(int(str(x.TDAYDATE)[:4]),int(str(x.TDAYDATE)[4:],10),x.TRAVDAY).date(), axis=1)
    df_trip['date_'] = pd.to_datetime(df_trip['TDAYDATE'], format='%Y%m') + pd.offsets.MonthBegin(0)
    df_trip['day_of_week'] = df_trip['TRAVDAY']

    df_trip["id"] = df_trip[['HOUSEID', 'PERSONID']].apply(lambda x: int(f"{x.HOUSEID}{x.PERSONID}"), axis=1)
    # df_trip = df_trip.groupby('id').filter(lambda x: len(x) > 1)

    df_trip = categorize_age(df_trip)
    df_trip['time_of_day'] = df_trip['STRTTIME'].apply(lambda x: ((x // 100) * 60 + (x % 100)) // 30 + 1)

    df_trip['WHYTRP1S'] = df_trip.WHYTRP1S.replace({10: 2, 40: 3, 20: 4, 30: 5, 50: 5, 70: 5, 80: 5, 97: 5})
    df_trip = df_trip[
        ["id", "R_RELAT", "HHFAMINC", "age_category", "R_SEX", "WORKER", "TRPTRANS", "day_of_week", "time_of_day", "date_", "WHYTRP1S"]]
    df_trip.drop_duplicates(inplace=True)
    dd = {1:20,2:18,3:1,4:3,5:2,6:4,7:21,8:7,9:21,10:9,11:8,12:17,13:21,14:21,15:13,16:11,17:15,18:21,19:14,20:21,97:21}
    df_trip['TRPTRANS'] = df_trip.TRPTRANS.replace({10: 11})
    df_trip = df_trip[df_trip.TRPTRANS.isin(dd.values())]
    df = df_trip.sort_values(['id', 'time_of_day'])
    return df


df = create_df()


# Aggregate data: count of each trip purpose per month
df_aggregated = df.groupby(['date_', 'WHYTRP1S']).size().reset_index(name='count')

# Let's say you want to forecast for each trip purpose separately
trip_purpose = 1  # Example for trip purpose = 1
df_purpose = df_aggregated[df_aggregated['WHYTRP1S'] == trip_purpose].copy()

# Rename columns for Prophet
# df_purpose.rename(columns={'date_': 'ds', 'count': 'y'}, inplace=True)

# Prepare additional regressors
# Aggregate additional features monthly
# For categorical features, use the mode (most frequent)
df_regressors = df.groupby('date_').agg({
    'HHFAMINC': 'mean',
    'R_SEX': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
    'age_category': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
    'WORKER': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
    'time_of_day': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
    'TRPTRANS': 'mean',
    'R_RELAT': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
    'day_of_week': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
}).reset_index()

# Merge with df_purpose
df_merged = pd.merge(df_purpose, df_regressors, on='date_', how='left')

# Rename 'date_' to 'ds' again if necessary
df_merged.rename(columns={'date_': 'ds','count': 'y'}, inplace=True)

# Handle categorical regressors using One-Hot Encoding
categorical_features = ['R_SEX', 'age_category', 'WORKER', 'time_of_day', 'R_RELAT', 'day_of_week']
df_encoded = pd.get_dummies(df_merged, columns=categorical_features, drop_first=True)

# List of regressors
regressors = [col for col in df_encoded.columns if col not in ['ds', 'y', 'WHYTRp1s']]

# Initialize Prophet
model = Prophet()

# Add regressors
for reg in regressors:
    model.add_regressor(reg)

# Fit the model
model.fit(df_encoded[['ds', 'y'] + regressors])

# Create a dataframe to hold future months
future_periods = 12  # Forecasting for next 12 months
future = model.make_future_dataframe(periods=future_periods, freq='M')

# Prepare future regressors
# Here, you need to provide future values for the regressors
# This can be based on historical patterns, domain knowledge, or assumptions
# For simplicity, we'll use the last available values
last_values = df_encoded[regressors].iloc[-1]

# Repeat last_values for each future period
future_regressors = pd.DataFrame([last_values.values] * len(future), columns=regressors)

# If you have trends or seasonality in regressors, consider modeling them separately

# Combine with future dates
future = pd.concat([future, future_regressors], axis=1)

# Predict
forecast = model.predict(future)

# Plot forecast
fig1 = model.plot(forecast)
plt.title(f'Forecast for Trip Purpose {trip_purpose}')
plt.xlabel('Date')
plt.ylabel('Count')
plt.show()

# Plot forecast components
fig2 = model.plot_components(forecast)
plt.show()
