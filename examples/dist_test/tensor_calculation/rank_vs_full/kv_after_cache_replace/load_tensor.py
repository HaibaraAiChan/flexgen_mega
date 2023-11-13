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



# tensor1= show_data('rank_0_v.pt')
# tensor2 = show_data('rank_1_v.pt')
# print()
# t = show_data('full_v.pt') 
# t_combine = torch.cat((tensor1, tensor2), dim=1)
# are_equal = torch.equal(t_combine, t)
# print('check pass ', are_equal)
# unequal_indices = torch.nonzero(t_combine != t)
# print('unequal indices ', unequal_indices)
# print(len(unequal_indices))

tensor1= show_data('rank_0_k.pt')[:-1]
tensor2 = show_data('rank_1_k.pt')[:-1]
print()
t = show_data('full_k.pt') [:-1]
print('shape of rank 0 ', tensor1.shape)
print('shape of rank 1 ', tensor2.shape)
print('shape of full k  ', t.shape)

t_combine =[]
# indices = [list(range(6)), list(range(12, 18)), list(range(24, 30)), list(range(36, 42))]
indices = [list(range(6)), list(range(6, 12)), list(range(12, 18)), list(range(18,24))]
for i in range(4):
    print("tensor1[indices[i]].shape ",tensor1[:,indices[i],:].shape)
    tmp = torch.cat((tensor1[:,indices[i],:], tensor2[:,indices[i],:]), dim=1)
    print('tmp.shape , ', tmp.shape)
    t_combine.append(tmp) 
    
print("befor combine t_combine.shape ", len(t_combine))
t_combine = torch.cat(t_combine, dim=1)
print('t_combine.shape ',t_combine.shape)
are_equal = torch.equal(t_combine, t)
print('check pass ', are_equal)
unequal_indices = torch.nonzero(t_combine != t)
print('unequal indices ', unequal_indices)
print(len(unequal_indices))




# tensor1= show_data('rank_0_k.pt')[-1]
# tensor2 = show_data('rank_1_k.pt')[-1]
# print()
# t = show_data('full_k.pt') [-1]
# t_combine = torch.cat((tensor1, tensor2), dim=1)
# are_equal = torch.equal(t_combine, t)
# print('check pass ', are_equal)
# unequal_indices = torch.nonzero(t_combine != t)
# print('unequal indices ', unequal_indices)
# print(len(unequal_indices))