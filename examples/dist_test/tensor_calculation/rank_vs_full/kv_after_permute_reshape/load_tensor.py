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
    return t[0]

tensor1= show_data('rank_0_k.pt')
tensor2 = show_data('rank_1_k.pt')
print()
t = show_data('full_k_new.pt') 
t_combine = torch.cat((tensor1, tensor2), dim=1)
are_equal = torch.equal(t_combine, t)
print('check pass ', are_equal)
unequal_indices = torch.nonzero(t_combine != t)
print('unequal indices ', unequal_indices)
print(len(unequal_indices))


# tensor10= show_data('rank_0_k.pt')[-1]
# tensor20 = show_data('rank_1_k.pt')[-1]


# t0= show_data('full_k_new.pt')[-1]
# # t_combine = torch.cat((tensor1, tensor2), dim=2)
# # are_equal = torch.equal(t_combine, t)
# # print('check pass ', are_equal)
# # unequal_indices = torch.nonzero(t_combine != t)
# # print('unequal indices ', unequal_indices)
# # print(len(unequal_indices))



