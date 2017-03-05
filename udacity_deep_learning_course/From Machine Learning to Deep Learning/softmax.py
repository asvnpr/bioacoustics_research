"""Softmax."""
import numpy as np


#scores = [3.0, 1.0, 0.2]
#scores = [1.0, 2.0, 3.0]
scores = np.array([[1, 2, 3, 6],[2, 4, 5, 6],[3, 8, 7, 6]])


def softmax(x):
    if (np.array(x).ndim == 1):
        #array of converted scores to return
        probs = []
        for i in range(0,len(x)):
            probs.append(np.exp(x[i]) / np.sum(np.exp(x)))
        #plotting requires the list/array to be a numpy array
        return np.array(probs)
    else:
        #array of converted scores to return
        probs = []
        #samples is 2d array. hold each sample(column) in 2d array
        samples = np.transpose(x)
        samples_rows = samples.shape[0]
        samples_cols = samples.shape[1]
        for i in range(0, samples_cols):
            probs_row = []
            for j in range(0, samples_rows):
                probs_row.append(np.exp(x[i][j]) / (np.sum(np.exp(samples[j]))))
            probs.append(probs_row)
        #plotting requires the list/array to be a numpy array
        return np.array(probs)




print(softmax(scores))


# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])


plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
