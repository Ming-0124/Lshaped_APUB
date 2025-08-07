import numpy as np

seed = 1234

epoch = 30

bootstrap_size = 1500   # M

data_size = 120  # N

n_items = 4   # x dimension
c = [-12, -20, -18, -40]

n_machines = 2  # y dimension
q = [8, 3]

b = np.zeros(n_machines)

A = np.zeros((n_machines, n_items))
