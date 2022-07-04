from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

# the architectures of origin paper
DARTS_V1 = Genotype(
    # the normal cell: have 2 input nodes(0,1), 4 intermediate nodes(2,3,4,5) and 1 output node(6)
    # for example:
    #   node 2
    #       has 2 edges
    #       the first edge is connected with node 1, and operation is sep_conv_3x3
    #       the second edge is connected with node 0, and operation is sep_conv_3x3
    #   node 3
    #       has 2 edges
    #       the first edge is connected with node 0, and operation is skip_connect
    #       the second edge is connected with node 1, and operation is sep_conv_3x3
    #   ......
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0),  # edges of node 2
            ('skip_connect', 0), ('sep_conv_3x3', 1),  # edges of node 3
            ('skip_connect', 0), ('sep_conv_3x3', 1),  # edges of node 4
            ('sep_conv_3x3', 0), ('skip_connect', 2)],  # edges of node 5
    # the outputs of node 2, 3, 4 and 5 are concatenated as the output of cell
    normal_concat=[2, 3, 4, 5],
    # the reduction cell
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('skip_connect', 2), ('max_pool_3x3', 0),
            ('max_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 2), ('avg_pool_3x3', 0)],
    reduce_concat=[2, 3, 4, 5])

DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('skip_connect', 0), ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('skip_connect', 2), ('max_pool_3x3', 1),
            ('max_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 2), ('max_pool_3x3', 1)],
    reduce_concat=[2, 3, 4, 5])

cifar10_evolution = Genotype(
    normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('dil_conv_5x5', 3), ('dil_conv_3x3', 2),
            ('dil_conv_5x5', 4), ('dil_conv_5x5', 3)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('max_pool_3x3', 2), ('skip_connect', 0),
            ('max_pool_3x3', 2), ('max_pool_3x3', 3),
            ('dil_conv_5x5', 4), ('dil_conv_5x5', 3)],
    reduce_concat=[2, 3, 4, 5])

cifar100_evolution = Genotype(
    normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 0), ('max_pool_3x3', 2),
            ('max_pool_3x3', 0), ('max_pool_3x3', 2),
            ('max_pool_3x3', 4), ('max_pool_3x3', 3)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0),
            ('max_pool_3x3', 2), ('max_pool_3x3', 0),
            ('max_pool_3x3', 2), ('max_pool_3x3', 0),
            ('max_pool_3x3', 2), ('dil_conv_5x5', 4)],
    reduce_concat=[2, 3, 4, 5])

mixture_evolution = Genotype(
    normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0),
            ('max_pool_3x3', 2), ('dil_conv_5x5', 1),
            ('max_pool_3x3', 2), ('max_pool_3x3', 3),
            ('max_pool_3x3', 3), ('max_pool_3x3', 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1),
            ('max_pool_3x3', 0), ('max_pool_3x3', 2),
            ('max_pool_3x3', 2), ('dil_conv_5x5', 0),
            ('max_pool_3x3', 2), ('dil_conv_5x5', 4)],
    reduce_concat=[2, 3, 4, 5])

mdenas_pmnist = Genotype(
    normal=[('max_pool_3x3', 0), ('avg_pool_3x3', 1),
            ('max_pool_3x3', 1), ('sep_conv_3x3', 0),
            ('avg_pool_3x3', 0), ('skip_connect', 3),
            ('dil_conv_3x3', 1), ('dil_conv_3x3', 0)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('dil_conv_3x3', 1), ('avg_pool_3x3', 0),
            ('max_pool_3x3', 2), ('dil_conv_5x5', 1),
            ('skip_connect', 2), ('avg_pool_3x3', 1),
            ('avg_pool_3x3', 3), ('sep_conv_3x3', 4)],
    reduce_concat=[2, 3, 4, 5]
)

mdenas_mixture = Genotype(
    normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1),
            ('avg_pool_3x3', 2), ('max_pool_3x3', 1),
            ('sep_conv_3x3', 2), ('dil_conv_3x3', 1),
            ('sep_conv_3x3', 1), ('avg_pool_3x3', 3)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('skip_connect', 2), ('avg_pool_3x3', 0),
            ('dil_conv_5x5', 2), ('dil_conv_3x3', 0),
            ('avg_pool_3x3', 0), ('skip_connect', 2)],
    reduce_concat=[2, 3, 4, 5]
)
