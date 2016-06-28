import numpy


values = numpy.load('../../resources/VGG_16_4ch.npy').item()
print values.keys()
for key in values:
    value = values[key]
    print 'key:{}, value:{}'.format(key, len(value['weights']))
