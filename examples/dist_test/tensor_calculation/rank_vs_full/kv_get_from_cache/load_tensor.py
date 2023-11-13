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

# tensor1= show_data('rank_0_k.pt')
# tensor2 = show_data('rank_1_k.pt')
# print()
# t = show_data('full_k.pt') 
# t_combine = torch.cat((tensor1, tensor2), dim=2)
# are_equal = torch.equal(t_combine, t)
# print('check pass ', are_equal)
# unequal_indices = torch.nonzero(t_combine != t)
# print('unequal indices ', unequal_indices)
# print(len(unequal_indices))

# tensor1= show_data('rank_0_q.pt')
# tensor2 = show_data('rank_1_q.pt')
# print()
# t = show_data('full_q.pt') 
# t_combine = torch.cat((tensor1, tensor2), dim=2)
# are_equal = torch.equal(t_combine, t)
# print('check pass ', are_equal)
# unequal_indices = torch.nonzero(t_combine != t)
# print('unequal indices ', unequal_indices)
# print(len(unequal_indices))

# tensor1= show_data('rank_0_v.pt')
# tensor2 = show_data('rank_1_v.pt')
# print()
# t = show_data('full_v.pt') 
# tensor1 = torch.round(tensor1 * 10000) / 10000
# tensor2 = torch.round(tensor2 * 10000) / 10000
# t = torch.round(t * 10000) / 10000
tensor10= show_data('rank_0_k.pt')[-1]
tensor20 = show_data('rank_1_k.pt')[-1]

t0= show_data('full_k.pt')[-1]
tnew0= show_data('full_k_new.pt')[-1]
are_equal_10 = torch.equal(t0, tnew0)

print('pass 0',are_equal_10)
unequal_indices = torch.nonzero(t0!=tnew0)
print('unequal indices 10', unequal_indices)
print(len(unequal_indices))

# print('tensor10 ', tensor10)
# print('tensor10.shape ', tensor10.shape)
# print('tensor20 ', tensor20)
# print('tensor20.shape ', tensor20.shape)

# print('t0 ', t0)
# print('t0.shape ', t0.shape)

# are_equal_10 = torch.equal(tensor10, t0)
# are_equal_20 = torch.equal(tensor20, t0)
# print('pass 10',are_equal_10)
# print('pass 20',are_equal_20)
# unequal_indices = torch.nonzero(tensor10 != t0)
# print('unequal indices 10', unequal_indices)
# print(len(unequal_indices))

# unequal_indices = torch.nonzero(tensor20 != t0)
# print('unequal indices 20', unequal_indices)
# print(len(unequal_indices))


