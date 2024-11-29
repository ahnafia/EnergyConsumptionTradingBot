import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import sklearn
from sklearn.metrics import mean_squared_error

df = pd.read_csv("PJME_hourly.csv")
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)
df.head()
df.index

df.plot(figsize=(10, 5), title="Hourly ENergy Usage (MB)")
plt.show()

train = df.loc[df.index < "01-01-2015"]
test = df.loc[df.index >= "01-01-2015"]


figure, axis = plt.subplots(figsize=(10, 5))
train.plot(ax=axis, label="Train")
test.plot(ax=axis, label="Test")

df.loc[(df.index > '01-01-2011') & (df.index < '01-08-2011')].plot()

def create_features(df):
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear
    return df

df = create_features(df)

figure, axis = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x='hour', y="PJME_MW")
plt.show()

figure, axis = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x='month', y="PJME_MW")
plt.show()

figure, axis = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x='dayofweek', y="PJME_MW")
plt.show()

figure, axis = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x='year', y="PJME_MW")
plt.show()

train = create_features(train)
test = create_features(test)


features = ['hour', 'dayofweek', 'quarter', 'month', 'year','dayofyear']
target = "PJME_MW"

x_train = train[features]
y_train = train[target]

x_test = test[features]
y_test = test[target]

reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, learning_rate=0.01)
reg.fit(x_train, y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        verbose=100)

pd.DataFrame(reg.feature_importances_, index=reg.feature_names_in_)

test['Predictions'] = reg.predict(x_test)
df = df.merge(test[['Predictions']], how='left', left_index=True, right_index=True)

axis = df[["PJME_MW"]].plot(figsize=(15,10))
df['Predictions'].plot(ax=axis)
plt.legend(["Real Amount", "Prections"])
plt.show()

axis = df.loc[(df.index > '04-01-2016') & (df.index < '04-08-2016')]['PJME_MW'].plot(figsize=(15, 5), title='Week Of Data')
df.loc[(df.index > '04-01-2016') & (df.index < '04-08-2016')]['Predictions'].plot(style='.')

score = np.sqrt(mean_squared_error(test['PJME_MW'], test['Predictions']))
print(f'RMSE: {score:0.2f}')

df = df.loc[df.index > "01-01-2015"]

df.loc[df.index >= "01-01-2015"].head(-20)

df_daily = (
        df.groupby(df.index.date)
        .agg({
            "Predictions": "mean",
            "up_or_down": "last"
        })
        .rename(columns={"Predictions": "average_consumption"})
    )


df_daily['next_day_usage'] = df_daily['average_consumption'].shift(-1)
df_daily["up_or_down"] = (df_daily['average_consumption'] < df_daily['next_day_usage']).astype(int)

df_daily.to_csv('hourly_predictions.csv', index=False)