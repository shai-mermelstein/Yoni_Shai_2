import module_name as module

import numpy as np

print("I'm in python")


arr = np.require([[1, 2.0], [3.3, 4.4]], dtype=np.float64, requirements=['C', 'A', 'W'])
print(arr)

res = module.func2(arr)

print("I'm in python")
print(arr)
print('sum =', res)