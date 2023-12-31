import torch
import torch.nn.functional as F

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
    return t

tensor1= show_data('rank_0_attn_weights.pt')
tensor2 = show_data('rank_1_attn_weights.pt')
print()
t = show_data('full_attn_weights.pt') 
t_combine = torch.cat((tensor1, tensor2), dim=1)
are_equal = torch.equal(t_combine, t)
print('check pass ', are_equal)
unequal_indices = torch.nonzero(t_combine != t)
print('unequal indices ', unequal_indices)
print(len(unequal_indices))
   
# tensor1= show_data('rank_0_value.pt')
# tensor2 = show_data('rank_1_value.pt')
# print()
# t = show_data('full_value.pt') 
# t_combine = torch.cat((tensor1, tensor2), dim=0)
# are_equal = torch.equal(t_combine, t)
# print('check pass ', are_equal)
# unequal_indices = torch.nonzero(t_combine != t)
# print('unequal indices ', unequal_indices)
# print(len(unequal_indices))
