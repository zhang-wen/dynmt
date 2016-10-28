import numpy as np
#from _gdynet import *
from _dynet import *

m = Model()
lp = m.add_lookup_parameters((1000, 10))

# Add parameters
W = parameter(m.add_parameters((5, 10)))
builder = LSTMBuilder(1, 10, 10, m)

trainer = AdamTrainer(m)
num_epochs = 10
minibatch_size = 3
seq_length = 5
train_set = [(1, 1, 1, 1, 1), (2, 2, 2, 2, 2), (1, 1, 1, 1, 1)]
labels = [1, 2, 1]

for epoch in range(num_epochs):

    total_loss = 0.0

    # Use the entire small training set as a batch, just for the example
    batch_indices = np.random.permutation(len(train_set))
    # print batch_indices

    # Batch lookup (for every timestamp of the LSTM)
    inputs = [lookup_batch(lp, [train_set[i][t] for i in range(
        minibatch_size)]) for t in range(seq_length)]
    # (5, 30) -> (5, 3, 10)
    for inp in inputs:
        print inp.npvalue() # (10, 3)
    #    print inp.npvalue().shape
    # print '...................'

    zw = builder.initial_state().transduce(inputs)
    # for z in zw:
    #    print z.npvalue().shape
    # Apply some transformations
    lstm_outputs = zw[-1]
    print lstm_outputs.npvalue().shape
    # reshape to (minibatch_size, 10)
    # reshape to (10, minibatch_size)
    lstm_outputs = reshape(lstm_outputs, (10,), batch_size=minibatch_size)
    print lstm_outputs.npvalue().shape
    output = W * lstm_outputs

    # Compute the loss for this batch
    curr_labels = [labels[i] for i in batch_indices]
    loss = pickneglogsoftmax_batch(output, curr_labels)

    sum_loss = sum_batches(loss)
    print loss.npvalue()
    print 'sum_loss => ', type(sum_loss), sum_loss.npvalue().shape
    print sum_loss.npvalue()

    total_loss += sum_loss.value()
    sum_loss.backward()
    trainer.update()

    trainer.update_epoch()
    total_loss /= len(labels)
    print 'Epoch', (epoch + 1), '/', num_epochs, 'Loss =', total_loss
