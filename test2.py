from dynet import *

m = Model()
lp = m.add_lookup_parameters((100, 10))

# Add parameters
W = parameter(m.add_parameters((5, 10)))
b = parameter(m.add_parameters((5)))
builder = LSTMBuilder(1, 10, 10, m)

# Batch lookup (for every timestamp of the LSTM)
batch = [(1, 2, 3, 4, 5), (5, 4, 3, 2, 1), (1, 1, 2, 2, 3)]
minibatch_size = 3
seq_length = 5
inputs = [lookup_batch(lp, [batch[i][t] for i in range(minibatch_size)])
          for t in range(seq_length)]

# 5 * (10, 3)
for i in inputs:
    print type(i)
    print i.npvalue()
    print i.npvalue().shape
# Apply some transformations
lstm_outputs = builder.initial_state().transduce(inputs)[-1]
# reshape to (minibatch_size, 10)
print 'lstm_outputs:'
print type(builder.initial_state().transduce(inputs))
print type(lstm_outputs)
print lstm_outputs.npvalue().shape
s = builder.initial_state().add_input(lstm_outputs)
print 's'
print s.h()[0].npvalue()
# (10, 3)
print reshape(lstm_outputs, (10,), batch_size=minibatch_size).npvalue().shape
lstm_outputs = transpose(
    reshape(lstm_outputs, (10,), batch_size=minibatch_size))
# (1, 10, 3)
print lstm_outputs.npvalue().shape
# (10, 1, 3)
print transpose(lstm_outputs).npvalue().shape
output = colwise_add(W * transpose(lstm_outputs), b)
# (5, 1, 3)

print 'output.npvalue().shape =', output.npvalue().shape, '(batch_size=3, output dimension=5)'

# Using pickneglogsoftmax_batch (fails with "what():  Bad input dimensions
# in PickNegLogSoftmax: [{5,3}]")
labels = [1, 2, 3]
print pickneglogsoftmax_batch(output, labels).npvalue().shape
# (1, 3)
print pickneglogsoftmax_batch(output, labels).npvalue()
