import torch

# 1.设置随机数种子
seed = 1024
torch.manual_seed(seed)

# 张量创建与属性查询

tensor_A = torch.randint(10, 100, [12]).reshape(3, 4)
print(tensor_A)  # 设置随机数种子后数据不会发生变化
print(f'tensor_A的形状:{tensor_A.shape}')
print(f'tensor_A的数据类型:{tensor_A.dtype}')
print(f'tensor_A的步长:{tensor_A.stride()}')
# 4:当我们在行上移动一个位置时,需要在底层内存中跨越4个元素,这是因为每一行包含4个元素,所以移动到下一行需要跳过整行
# 1:当我们在列上移动一个位置时,需要在底层内存中跨越1个元素,这是因为同一行中相邻的元素在内存中时相邻存储的

tensor_B = tensor_A[1:3, 1:3]
'''
tensor([[51, 11],
        [26, 45]])
'''
print(f'tensor_B的偏移量:{tensor_B.storage_offset()}')  # 5
'''
tensor_B 的第一个元素是 51 对应 tensor_A[1,1]
tensor_A 是 3行4列，按行优先存储，第 0 行占 0-3 索引，第 1 行从索引 4 开始
tensor_A[1,1] 是第 1 行的第 1 列
索引 = 第 1 行起始索引（4） + 列索引（1） = 5

'''
tensor_C = tensor_A.transpose(0, 1)
print(f'tensor_C是否连续:{tensor_C.is_contiguous()}')

tensor_C = tensor_C.contiguous().flatten()
print(tensor_C)
