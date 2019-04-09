from core.data_processor import DataLoader
from core.model import Model
import pandas as pd
import json

date=input("input a date in the format as 2017-12-13: ")
stockID=input("input a stockID: ")

configs = json.load(open('config.json', 'r'))
print("reading training data...")
dataframe = pd.read_csv(configs['data']['filename'])
dataframe = dataframe[dataframe['id'] == int(stockID)]
dataframe.reset_index(drop=True, inplace=True)
index = dataframe[dataframe['time'] == date].index.values.astype(int)[0]
dataframe_window = dataframe[index-49:index+2]
print("making stock " + stockID + "'s dataframe")

data = DataLoader(
    dataframe_window,
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
print(predictions)

curr_price = x_test[0][-1][0]
print("curr price: ", curr_price)
changes = []
for p in predictions[0]:
    print((p+1) / (curr_price+1) -1)
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

dataframe_true = dataframe[index:index+6]
prices = dataframe_true.get('adjusted_close').values
print("true prices:", prices)
true_changes = prices[1:6]/prices[0]-1
print("true changes: ", max(true_changes), min(true_changes))

# for p in predictions[0]:
#     change = abs(p / curr_price - 1)
#     if change >= 0.02:
#         if p > curr_price:
#             if change >= 0.05:
#                 print("stockID: " + stockID + ", prediction time: " + date + ", change: +5%\n")
#                 print("stockID: " + stockID + ", prediction time: " + date + ", change: +3%\n")
#                 print("stockID: " + stockID + ", prediction time: " + date + ", change: +2%\n")
#             elif change >= 0.03:
#                 print("stockID: " + stockID + ", prediction time: " + date + ", change: +3%\n")
#                 print("stockID: " + stockID + ", prediction time: " + date + ", change: +2%\n")
#             else:
#                 print("stockID: " + stockID + ", prediction time: " + date + ", change: +2%\n")
#         else:
#             if change >= 0.05:
#                 print("stockID: " + stockID + ", prediction time: " + date + ", change: -5%\n")
#                 print("stockID: " + stockID + ", prediction time: " + date + ", change: -3%\n")
#                 print("stockID: " + stockID + ", prediction time: " + date + ", change: -2%\n")
#             elif change >= 0.03:
#                 print("stockID: " + stockID + ", prediction time: " + date + ", change: -3%\n")
#                 print("stockID: " + stockID + ", prediction time: " + date + ", change: -2%\n")
#             else:
#                 print("stockID: " + stockID + ", prediction time: " + date + ", change: -2%\n")
#         break
