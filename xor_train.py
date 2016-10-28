from dynet import *

# define the parameters
m = Model()
pW = m.add_parameters((8, 2))
pV = m.add_parameters((1, 8))
pb = m.add_parameters((8, 1))

# renew the computation graph
renew_cg()

# add the parameters to the graph
W = parameter(pW)
V = parameter(pV)
b = parameter(pb)

# create the network
x = vecInput(2)  # an input vector of size 2.
output = logistic(V * (tanh((W * x) + b)))
# define the loss with respect to an output y.
y = scalarInput(0)  # this will hold the correct answer
loss = binary_log_loss(output, y)

# create training instances


def create_xor_instances(num_rounds=2000):
    questions = []
    answers = []
    for round in xrange(num_rounds):
        for x1 in 0, 1:
            for x2 in 0, 1:
                answer = 0 if x1 == x2 else 1
                questions.append((x1, x2))
                answers.append(answer)
    return questions, answers

questions, answers = create_xor_instances()
# print questions
# print answers
# [(0, 0), (0, 1), (1, 0), (1, 1), (0, 0), (0, 1), (1, 0), (1, 1)]
# [0, 1, 1, 0, 0, 1, 1, 0]

# train the network
trainer = SimpleSGDTrainer(m)

total_loss = 0
seen_instances = 0
for question, answer in zip(questions, answers):
    x.set(question)
    y.set(answer)
    seen_instances += 1
    total_loss += loss.value()
    loss.backward()
    trainer.update()
    #print 'W', W.value()
    #print 'b', b.value()
    if (seen_instances > 1 and seen_instances % 100 == 0):
        print 'seen ', seen_instances, ' instances'
        print "average loss is:", total_loss / seen_instances

x.set([0, 1])
print '0, 1', output.value()
x.set([1, 0])
print '1, 0', output.value()
x.set([0, 0])
print '0, 0', output.value()
x.set([1, 1])
print '1, 1', output.value()
