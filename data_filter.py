import matplotlib.pyplot as plt
import csv
import numpy as np


steering = []
with open("data/driving_log.csv") as f, open("data/new_driving_log.csv", 'w') as filtered:
	reader = csv.reader(f, delimiter=',')
	
	for rows in reader:
		if float(rows[3]) <= 0.15:
			prob = np.random.uniform()
			if prob > 0.8:
				rows = ','.join(rows)
				filtered.write(rows)
				filtered.write("\n")
		else:
			rows = ','.join(rows)
			filtered.write(rows)
			filtered.write("\n")
