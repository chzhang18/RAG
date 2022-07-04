from collections import namedtuple

#Genotype = namedtuple('Genotype_2D', 'cell cell_concat')
Genotype = namedtuple('Genotype_2D', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'skip_connect_2d',
    'conv_3x3']


