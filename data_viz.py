import matplotlib.pyplot as plt
import csv
import numpy as np


steering = []
with open("data/udacity_driving_log_enhanced.csv") as f:
	reader = csv.reader(f, delimiter=',')
	for rows in reader:
		if(abs(float(rows[3].strip()))) <= 0.01:
			prob = np.random.random()
			if prob > 0.2:
				steering.append(float(rows[3].strip()))
		else:
			steering.append(float(rows[3].strip()))
		
angles = np.unique(steering)
n, bins, patches = plt.hist(steering, bins=30, normed=0)
print(bins)
print("Unique num of angles = {}".format(len(angles)))
print("Number of samples = {}".format(len(steering)))
plt.show()
