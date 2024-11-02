from load_data import load_data
from parameter import parse_args
import numpy as np

data_document = np.load('train.npy', allow_pickle=True).item()

args = parse_args()  # load parameters
train_data, dev_data, test_data = load_data(args)

def replaceNode(data):
    for i in range(len(data)):
        data[i]['node'][-1] = "[MASK], 这是你们需要预测的事件，从'candiSet'中选择"
        temp_candi = [event[5] for event in data[i]['candiSet']]
        data[i]['candiSet'] = temp_candi
    return data

train_data = replaceNode(train_data)
dev_data = replaceNode(dev_data)
test_data = replaceNode(test_data)

for i in range(len(test_data)):
    test_data[i]['label'] = "None, 这是你们需要预测的事件，表示在candiSet中的位置索引"

np.save('train4.npy', {"train_data": train_data, "dev_data": dev_data, "test_data": test_data})

print(111)