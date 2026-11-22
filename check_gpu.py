"""Actual CUDA forward/backward checks for every project model."""
import torch
import torch_geometric
from train import Model, seed_everything
print('PyTorch:',torch.__version__,'CUDA runtime:',torch.version.cuda,'PyG:',torch_geometric.__version__)
assert torch.cuda.is_available(), 'CUDA unavailable. Send nvidia-smi output and this error.'
print('GPU:',torch.cuda.get_device_name(0),'Compute capability:',torch.cuda.get_device_capability(0))
print('Wheel architecture list:',torch.cuda.get_arch_list())
torch.set_num_threads(2)
seed_everything(42)
x=torch.randn(32,16,device='cuda')
a=torch.arange(32,device='cuda')
e=torch.stack([torch.cat([a,a]),torch.cat([(a+1)%32,(a-1)%32])])
y=torch.arange(32,device='cuda')%7
for name in ['mlp','gcn','gat','gated','fixed']:
 model=Model(name,16,7).cuda()
 opt=torch.optim.Adam(model.parameters(),lr=.01)
 opt.zero_grad()
 loss=torch.nn.functional.cross_entropy(model(x,e),y)
 loss.backward()
 assert torch.isfinite(loss).item()
 assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
 opt.step()
 torch.cuda.synchronize()
 print(name,'CUDA forward/backward/optimizer: PASS')
print('GPU READY. This synthetic check tests CUDA operations; it is NOT a Cora experiment.')
