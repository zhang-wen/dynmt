from _gdynet import *

renew_cg()

m = Model()
trainer = AdamTrainer(m)
builder = LSTMBuilder(1, 10, 10, m)

s0 = builder.initial_state()
s1 = s0.add_input(vecInput(10,))

print 's1'
print s1.s()[0].npvalue().shape
print s1.s()[0].npvalue()

print s1.s()[1].npvalue().shape
print s1.s()[1].npvalue()

output1 = s1.output()

print 'output1'
print output1.npvalue().shape
print output1.npvalue()

s2 = s1.add_input(output1)  # no prblem

print 'old s2'
print s2.s()[0].npvalue().shape
print s2.s()[0].npvalue()

output2 = s2.output()

print 'old output2'
print output2.npvalue().shape
print output2.npvalue()

ws = parameter(m.add_parameters((10, 10)))
bs = parameter(m.add_parameters((10)))
new_s = ws * s1.h()[-1] + bs     # problem here ...

print 'new_s ...'
print type(new_s)
print new_s.npvalue().shape
print new_s.npvalue()

print 'before ...'
print s1.h()[0].npvalue().shape
print s1.h()[0].npvalue()

print 's1.s()...'
for i in s1.s():
    print i.npvalue()

print 'type: ', type(new_s)
ss1 = s1.set_h((new_s,))  # need return

'''
print 'ss1.s()...'
for i in ss1.s():
    print i.npvalue()
'''
print type(ss1)

s2 = ss1.add_input(ss1.output())

print 'new s2'
print s2.h()[0].npvalue().shape
print s2.h()[0].npvalue()

output2 = s2.output()

print 'new output2'
print output2.npvalue().shape
print output2.npvalue()

aa = parameter(m.add_parameters((2, 3)))
print 'aa'
print aa.npvalue().shape
print aa.npvalue()
aaa = reshape(aa, (aa.npvalue().shape[0] * aa.npvalue().shape[1],))
print 'aaa'
print aaa.npvalue().shape
print aaa.npvalue()

dot = dot_product(aaa, aaa)
print 'dot'
print dot.npvalue().shape
print dot.npvalue()

dotmul = cwise_multiply(aa, aa)
print 'dotmul'
print dotmul.npvalue().shape
print dotmul.npvalue()
