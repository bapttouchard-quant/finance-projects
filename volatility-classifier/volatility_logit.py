import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score

returns = yf.download("SPY", start= "2000-01-01", end= "2026-01-01", auto_adjust=True)["Close"].pct_change().squeeze().dropna()

vol_5 = returns.rolling(5).std()
vol_21 = returns.rolling(21).std()
ret_5 = returns.rolling(5).sum()
future_vol_5 = returns.rolling(5).std().shift(-5)

target = np.where(future_vol_5 > vol_21, 1, 0)

features = pd.DataFrame({"vol_5": vol_5, "vol_21": vol_21, "ret_5": ret_5, "target": target}).dropna().iloc[:-5]
features["target"].value_counts(normalize = True)

X = features.drop(columns="target")
y = features["target"]

X_train = X.loc[:"2018-12-31"]
X_test = X.loc["2019-01-01":]
y_train = y.loc[:"2018-12-31"]
y_test = y.loc["2019-01-01":]

most_common = y_train.mode()[0]

benchmark_1 = (y_test == most_common)
benchmark_1 = benchmark_1.mean()

rule_predictions = np.where(X_test["vol_5"] > X_test["vol_21"], 1, 0)

benchmark_2 = (rule_predictions == y_test)
benchmark_2 = benchmark_2.mean()

model = LogisticRegression()
model.fit(X_train, y_train)

logit_prediction = model.predict(X_test)
check = (logit_prediction == y_test).mean()

#___________________________________________

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#print(X_train_scaled, X_test_scaled)

model_scaled = LogisticRegression()
model_scaled.fit(X_train_scaled, y_train)

predictions_scaled = model_scaled.predict(X_test_scaled)

accuracy = (predictions_scaled == y_test).mean().round(3)

#print(confusion_matrix(y_test, predictions_scaled))

probabilities = model_scaled.predict_proba(X_test_scaled)[:, 1]

for threshold in [0.2, 0.3, 0.4, 0.5]:
    threshold_predictions = np.where(probabilities > threshold, 1, 0)

    recall = recall_score(y_test, threshold_predictions)
    precision = precision_score(y_test, threshold_predictions)
    accuracy = (threshold_predictions == y_test).mean()

    print(f"Threshold {threshold}: recall {recall:.1%}, precision {precision:.1%}, accuracy {accuracy:.1%}")

#print("Share of more Volatile", predictions_scaled.mean().round(3), "Accuracy", accuracy)
#print(logit_prediction.mean()) #,check)
#print("Simple Rule:", benchmark_2.round(3))
#print(rule_predictions)
#print(benchmark_1)
#print(most_common)
#print(X_train.index.max(), X_test.index.min())
#print(len(X_train), len(X_test))
#print(features.tail(10), features["target"].value_counts(normalize=True))
