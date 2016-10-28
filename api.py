#from _gdynet import *
from dynet import *

# ==== Create a new computation graph
# it is a singleton, we have one at each stage

# renew_cg() clears the current one and starts a new one
renew_cg()

# ==== Creating Expressions from user input / constants
x = scalarInput(3)
v = vecInput(3)
v.set([1, 2, 3])

z = matInput(2, 3)
z1 = matInput(2, 2)
z1.set([1, 2, 3, 4])

print z.value()
print z.npvalue()
print v.vec_value()
print x.scalar_value()
print x.value()
