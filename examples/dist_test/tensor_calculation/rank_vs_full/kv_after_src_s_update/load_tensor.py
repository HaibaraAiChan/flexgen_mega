import torch
import torch.nn.functional as F
import numpy as np

def read_data(f1,f2):
    t1= torch.load(f1)
    t2= torch.load(f2)
    return t1,t2

def show_data(f):
    t = torch.load(f)
    print(f)
    print(f+'.shape',t.shape)
    print(f,t)
    return t
    
def show_data_0(f):
    t = torch.load(f)
    print(f)
    print(f+'.shape[0]',t.shape[0])
    print(f+'[0]')
    print(t[0])
    return t[0]

# tensor1= show_data('rank_0_k.pt')[:-1]
# tensor2 = show_data('rank_1_k.pt')[:-1]
# print()
# t = show_data('full_k.pt') [:-1]
# t_combine = torch.cat((tensor1, tensor2), dim=1)
# are_equal = torch.equal(t_combine, t)
# print('check pass ', are_equal)
# unequal_indices = torch.nonzero(t_combine != t)
# print('unequal indices ', unequal_indices)
# print(len(unequal_indices))

tensor1= show_data('rank_0_k.pt')[-1]
tensor2 = show_data('rank_1_k.pt')[-1]

t = show_data('full_k.pt') [-1]

tensor1 = torch.tensor(np.round(tensor1.numpy(), decimals=4))
tensor2 = torch.tensor(np.round(tensor2.numpy(), decimals=4))
t = torch.tensor(np.round(t.numpy(), decimals=4))
torch.set_printoptions(sci_mode=True)
print('tensor1 ', tensor1.half())
print('tensor2 ', tensor2.half())
print('t ', t.half())

t_combine = torch.cat((tensor1, tensor2), dim=0)
are_equal = torch.equal(t_combine, t)
print('check pass ', are_equal)
unequal_indices = torch.nonzero(t_combine != t)
print('unequal indices ', unequal_indices)
print(len(unequal_indices))





tensor1= show_data('rank_0_k.pt')[-1]
tensor2 = show_data('rank_1_k.pt')[-1]

t = show_data('full_k.pt') [-1]

tensor1 = torch.tensor(np.round(tensor1.numpy(), decimals=4))
tensor2 = torch.tensor(np.round(tensor2.numpy(), decimals=4))
t = torch.tensor(np.round(t.numpy(), decimals=4))
torch.set_printoptions(sci_mode=True)
print('tensor1 ', tensor1.half())
print('tensor2 ', tensor2.half())
print('t ', t.half())

t_combine = torch.cat((tensor1, tensor2), dim=0)
print('t_combine [6]', t_combine[6])
print('t [6]', t[6])
are_equal = torch.equal(t_combine[6], t[6])
print('check pass ', are_equal)
unequal_indices = torch.nonzero(t_combine[6] != t[6])
print('unequal indices ', unequal_indices)
print(len(unequal_indices))
# are_equal = torch.equal(t_combine, t)
# print('check pass ', are_equal)
# unequal_indices = torch.nonzero(t_combine != t)
# print('unequal indices ', unequal_indices)
# print(len(unequal_indices))