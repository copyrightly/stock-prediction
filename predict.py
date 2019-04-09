from core.data_processor import DataLoader
from core.model import Model
import pandas as pd
import json

date=input("input a date in the format as 2017-12-13: ")
stockID=input("input a stockID: ")
train_directory=input("input train directory: ")
test_directory=input("input test directory: ")

print("reading training dataset...")
train_dataframe = pd.read_csv(train_directory)
train_dataframe= train_dataframe[train_dataframe['id'] == int(stockID)]
train_dataframe.reset_index(drop=True, inplace=True)

print("reading test dataset...")
test_dataframe = pd.read_csv(test_directory)
test_dataframe = test_dataframe[test_dataframe['id'] == int(stockID)]
test_dataframe.reset_index(drop=True, inplace=True)

dataframe = pd.concat([train_dataframe, test_dataframe], ignore_index=True)

# dataframe.reset_index(drop=True, inplace=True)
index = dataframe[dataframe['time'] == date].index.values.astype(int)[0]
dataframe = dataframe[index-49:index+2]
print("making stock " + stockID + "'s dataframe")

configs = json.load(open('config.json', 'r'))
data = DataLoader(
    dataframe,
    int(stockID),
    0,
    configs['data']['columns']
)

x_test, y_test = data.get_test_data(
    seq_len=configs['data']['sequence_length'],
    normalise=configs['data']['normalise']
)
print("len of test(expected: 1):", len(x_test))

print("loading model...")
model = Model()
model.load_model('saved_models/stock-'+stockID+'.h5')

predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], 5)
# print("predictions: ", predictions)

curr_price = x_test[0][-1][0]
changes = []
for p in predictions[0]:
    # print("change: ", (p+1) / (curr_price+1) -1)
    changes.append((p+1) / (curr_price+1) -1)
if max(changes) >= 0.05:
    print("stockID: " + stockID + ", prediction time: " + date + ", change: +5%\n")
    print("stockID: " + stockID + ", prediction time: " + date + ", change: +3%\n")
    print("stockID: " + stockID + ", prediction time: " + date + ", change: +2%\n")
elif max(changes) >= 0.03:
    print("stockID: " + stockID + ", prediction time: " + date + ", change: +3%\n")
    print("stockID: " + stockID + ", prediction time: " + date + ", change: +2%\n")
elif max(changes) >= 0.02:
    print("stockID: " + stockID + ", prediction time: " + date + ", change: +2%\n")
if min(changes) <= -0.05:
     print("stockID: " + stockID + ", prediction time: " + date + ", change: -5%\n")
     print("stockID: " + stockID + ", prediction time: " + date + ", change: -3%\n")
     print("stockID: " + stockID + ", prediction time: " + date + ", change: -2%\n")
elif min(changes) <= -0.03:
     print("stockID: " + stockID + ", prediction time: " + date + ", change: -3%\n")
     print("stockID: " + stockID + ", prediction time: " + date + ", change: -2%\n")
elif min(changes) <= -0.02:
     print("stockID: " + stockID + ", prediction time: " + date + ", change: -2%\n")
