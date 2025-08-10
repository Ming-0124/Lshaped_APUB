import numpy as np

seed = 1234

epoch = 30

p = 0.1

mu1, var1 = 20, 5.76
mu2, var2 = 2, 0.16

bootstrap_size = 1500   # M

data_size = 120  # N

n_items = 4   # x dimension
c = [-12, -18, -20, -40]

n_machines = 2  # y dimension
q = [18, 9]

b = np.zeros(n_machines)

A = np.zeros((n_machines, n_items))
