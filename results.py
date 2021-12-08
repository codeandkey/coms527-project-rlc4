import matplotlib
import matplotlib.pyplot as plt
import json
import numpy as np

eval_data = None
loss_data = None

loss_x, loss_y = [], []
eval_x, eval_y = [], []

with open('eval', 'r') as f:
    eval_data = json.load(f)

with open('loss', 'r') as f:
    loss_data = json.load(f)

figure, axis = plt.subplots(2, 1)

for d in loss_data:
    loss_x.append(d[0])
    loss_y.append(d[1])

for d in eval_data:
    eval_x.append(d[0])
    eval_y.append(d[1])

axis[0].set_title('Loss average by generation')
axis[0].plot(loss_x, loss_y)

axis[1].set_title('Evaluation score by generation')
axis[1].plot(eval_x, eval_y, 'r')
#axis[1].set_ylim(0, 1)

plt.savefig('results.png')
plt.show()
