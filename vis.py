import pandas as pd
import matplotlib.pyplot as plt

# load the file
data = pd.read_csv("data.csv")

# plot the data
plt.plot(data['x'], data['y'], 'o')
plt.title('LR Predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.show()