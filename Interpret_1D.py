import numpy as np
import matplotlib.pyplot as plt


directory = 'D:/Research/Bin Optimization/SNR Values/'
mats = ['Au', 'Dy', 'Lu', 'Gd', 'I']

for i in np.arange(5):
    average = np.empty(7)
    n = 0
    for j in np.array([0.25, 0.5, 1, 2, 3, 4, 5]):
        x = np.load(directory + mats[i] + '_' + str(j) + 'percent.npy')
        s = x[0]
        w = x[1]
        idx = np.argmax(s) # 0:42
        average[n] = w[idx]
        n += 1

    print(mats[i])
    print(average)
    print(np.mean(average))
    print()

#%%
import numpy as np
import matplotlib.pyplot as plt


directory = 'D:/Research/Bin Optimization/SNR Values/'
mats = ['Au', 'Dy', 'Lu', 'Gd', 'I']
idx = 0

x = np.load(directory + mats[idx] + '_' + '3.0percent.npy')
s = x[0]
w = x[1]
i = np.argmax(s[0:42])
s = np.divide(s, s[i])
ow = '%0.3f' % w[i]

plt.figure(figsize=(11, 8))
plt.plot(w, s)
plt.plot(np.ones(100)*w[i], np.linspace(0, 1.2, 100))
plt.title(mats[idx] + ', 3 Percent Concentration', fontsize=40)
plt.xlabel('Bin Width (keV)', fontsize=30)
plt.ylabel('Normalized SNR', fontsize=30)
plt.tick_params(labelsize=25)
plt.annotate('Optimal Bin Width: ' + ow + ' keV', xy=(1, 0), xycoords='axes fraction',
             textcoords='offset pixels', xytext=(-10, 10), horizontalalignment='right', verticalalignment='bottom',
             fontsize=25)
plt.ylim([0, 1.2])
plt.show()
