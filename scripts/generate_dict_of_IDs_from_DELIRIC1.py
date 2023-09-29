import pandas as pd
import pickle

csv_file = "DELIRIC1_ID1_GNEprop.csv"

result_list = []

df = pd.read_csv(csv_file)

for idx, value in enumerate(df["ID1"]):
    result_list.append((value, 1))

csv_file1 = "DELIRIC1_ID2_GNEprop.csv"

df1 = pd.read_csv(csv_file1)

for idx, value in enumerate(df1["ID2"]):
    result_list.append((value, 2))

csv_file2 = "DELIRIC1_ID3_GNEprop.csv"

df2 = pd.read_csv(csv_file2)

for idx, value in enumerate(df2["ID3"]):
    result_list.append((value, 3))


print(result_list)
print("Length of the list of tuples:", len(result_list))

result_dict = {}

for i,tpl in enumerate(result_list):
    result_dict[i] = tpl

print(result_dict)
print("Length of the dictionary:", len(result_dict))

with open("dict_IDs_with_columns.pkl", "wb") as file:
    pickle.dump(result_dict, file)

print("Dictionary saved to dict_IDs_with_columns.pkl")