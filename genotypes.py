from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal_stage1 normal_stage2 normal_stage3 normal_concat reduce reduce_concat')

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


newGen = Genotype(normal_stage1=[('dil_conv_3x3', 1), ('dil_conv_5x5', 0), 
								('dil_conv_5x5', 0), ('dil_conv_5x5', 1), 
								('sep_conv_5x5', 0), ('sep_conv_5x5', 1), 
								('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], 
								normal_stage2=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), 
								('sep_conv_5x5', 0), ('sep_conv_3x3', 1), 
								('avg_pool_3x3', 0), ('avg_pool_3x3', 1), 
								('sep_conv_5x5', 0), ('avg_pool_3x3', 2)], 
								normal_stage3=[('dil_conv_3x3', 1), ('dil_conv_3x3', 0), 
								('dil_conv_5x5', 0), ('sep_conv_3x3', 1), 
								('dil_conv_5x5', 0), ('dil_conv_3x3', 1), 
								('dil_conv_5x5', 0), ('dil_conv_5x5', 2)], 
								normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), 
								('max_pool_3x3', 1), ('sep_conv_3x3', 2), 
								('max_pool_3x3', 1), ('sep_conv_3x3', 3), 
								('dil_conv_5x5', 2), ('dil_conv_5x5', 3)], 
								reduce_concat=range(2, 6))

