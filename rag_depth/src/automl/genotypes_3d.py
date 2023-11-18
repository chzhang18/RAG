from collections import namedtuple

#Genotype = namedtuple('Genotype', 'cell cell_concat')
Genotype_3D = namedtuple('Genotype', 'reduce reduce_concat')

PRIMITIVES_3D = [
    'skip_connect_3d',
    '3d_conv_3x3'
]

