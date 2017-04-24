import matplotlib.pyplot as plt
import csv
import numpy as np


steering = []
with open("../data/udacity_driving_log_with_correction.csv") as f:
	reader = csv.reader(f, delimiter=',')
	for rows in reader:
		# print(rows[0])
		if float(rows[6].strip()) > 10:
			if float(rows[3].strip()) >= 0.0 and float(rows[3].strip()) <= 0.08:
				prob = np.random.random()
				if prob >= 0.75:
					steering.append(float(rows[3].strip()))
			elif abs(float(rows[3].strip())) > 0.25 and abs(float(rows[3].strip())) < 0.5:
				steering.append(float(rows[3].strip()))
				steering.append(float(rows[3].strip()))
				steering.append(float(rows[3].strip()))
			elif abs(float(rows[3].strip())) >= 0.5:
				steering.append(float(rows[3].strip()))
				steering.append(float(rows[3].strip()))
				steering.append(float(rows[3].strip()))
				steering.append(float(rows[3].strip()))
				# steering.append(float(rows[3].strip()))
			else:
				steering.append(float(rows[3].strip()))

		
angles = np.unique(steering)
n, bins, patches = plt.hist(steering, bins=20, normed=0)
print(bins)
print("Unique num of angles = {}".format(len(angles)))
print("Number of samples = {}".format(len(steering)))
plt.show()

