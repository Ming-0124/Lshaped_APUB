import numpy as np

epoch = 30

p = 0.995

bootstrap_size = 1000   # M

data_size = 120  # N

n_items = 4   # x dimension
c = [-14, -9, -20, -15]

n_machines = 2  # y dimension

b = np.zeros(n_machines)

A = np.zeros((n_machines, n_items))
