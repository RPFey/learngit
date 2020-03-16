# RNN

对信息的理解可以分为时域， x(t), x(t+1), ... ; 或者空域(图片), I(row), I(row+1), ... 

定义 ：
```python
TIME_STEP = 28
# each input size (like a row in a image)
INPUT_SIZE = 28
```

网络(分类)
```python
self.rnn = nn.LSTM(
    input_size 
    hidden_size  # rnn 中的隐藏层, 也是输出维度
    num_layers
    batch_first  # 确定batch 维度  (batch_size, time_step,input_size)   
)
self.out = nn.Linear(input_size, output_size)

r_out, (h_n, h_c) = self.rnn(x, None) 
# (h_n, h_c) "主线的记忆", None 初始时刻的hidden state
out = self.out(r_out[:,-1,:])  # r_out (batch, time_step, input_size) 
```
相当于一次输入中 (time_step, input_size) 通过 rnn 后自动调用。

而下一个 batch_sample 时上一次的记忆则无效(没有采用hidden_state)

回归
```python
# 网络相同
def forward(self,x,hidden_state):
    r_out, hidden_state = self.rnn(x, hidden_state) # hidden_state 不断更新
    # hidden_state (n_layers, batch, hidden_size)
    outs = []    # save all predictions

    for time_step in range(r_out.size(1)):    # calculate output for each time step
        outs.append(self.out(r_out[:, time_step, :]))
    return torch.stack(outs, dim=1), h_state
```

since it's regression, it returns the prediction at each time step

train 
```python
h_state = None      # for initial hidden state

for step in range(100):
    prediction, h_state = rnn(x, h_state)   # rnn output
    # !! next step is important !!
    h_state = h_state.data        # repack the hidden state, break the connection from last iteration
```
不断更新 hidden_state

# encoder - decoder 

本质上信息的压缩与解压，利用压缩的信息训练神经网络

decoder 涉及到 反卷积之类的操作

# Auto-grad

尽量减少对矩阵的直接赋值(其实赋值的话根本没办法算梯度)，最好是生成一个常数矩阵，与原矩阵做加减乘除运算达到目的(比如筛选)

optimizer 的 zero_grad 只需要运行一次即可

自定义操作：

```python
class BinarizedF(Function):
    @staticmethod
    def forward(ctx, input_):  # 注意这里 ctx  的使用
        ctx.save_for_backward(input_)
        a = torch.ones_like(input_)
        b = -torch.ones_like(input_)
        output = torch.where(input_ >= 0, a, b)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        input_, = ctx.saved_tensors
        input_abs = torch.abs(input_)
        ones = torch.ones_like(input_)
        zeros = torch.zeros_like(input_)
        input_grad = torch.where(input_abs <= 1, ones, zeros)
        return input_grad

class Binarized(torch.nn.Module):
    def __init__(self):
        super(Binarized, self).__init__()

    def forward(self, input_):
        return BinarizedF.apply(input_)
```

自定义处理层，实现backward 处理。 从auto_grad 中继承 Function 实现 forward & backward

再继承 nn.Module， 实现 forward, 就与一般的神经网络层一致了

# Multi-GPU 
采用 "cuda:%d" 方法指定GPU
```python
cuda1 = torch.device("cuda:1")
data = data.to(cuda1)
```

多线程：(具体教程见<https://pytorch.org/tutorials/intermediate/dist_tuto.html> 和 <https://mp.weixin.qq.com/s/hVcgcMYf9AaCHJ_2F-VyZQ> )
1. pytorch.nn.distributed
```python
# setup environment
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(42)
# clean up environment
def cleanup():
    dist.destroy_process_group()
# 具体的训练过程， world_size 是总共的线程数
def train(rank, world_size):
    setup(rank, world_size)
    ...
    cleanup()
# main 中调用
def run_multi(fn, world_size):
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
   run_multi(train, 4)

# 还有一个进程间通信的方法
def average_gradient(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce_multigpu([param.grad.data], op=dist.ReduceOp.SUM)  # 将所有进程中 Tensor 求和并存储于个进程中
        param.grad.data /= size
```
save and load : 初始化的时候，将一个线程中的模型随机初始化并保存，然后其余线程导入这个模型

保存时，只需要保存一个线程中的模型 (一般为 rank 0)

多进程概念：

group : 进程组， 也是一个 world , 可以采用 new_group 接口创建 world 的子集

local_rank : 每个进程内 GPU 编号, 由 torch.distributed.launch 决定

流程 ：

1. 在使用 distributed 包的任何其他函数之前，需要使用 init_process_group 初始化进程组，同时初始化 distributed 包。

2. 如果需要进行小组内集体通信，用 new_group 创建子分组

3. 创建分布式并行模型 DDP(model, device_ids=device_ids)

4. 为数据集创建 Sampler

5. 使用启动工具 torch.distributed.launch 在每个主机上执行一次脚本，开始训练 (optional)

6. 使用 destory_process_group() 销毁进程组

## TCP 初始化

不同主机上多 GPU 训练(TCP 方式)：
```python
import torch.distributed as dist
import torch.utils.data.distributed

# ......
parser = argparse.ArgumentParser(description='PyTorch distributed training on cifar-10')
parser.add_argument('--rank', default=0,
                    help='rank of current process')
parser.add_argument('--word_size', default=2,
                    help="word size")
parser.add_argument('--init_method', default='tcp://127.0.0.1:23456',
                    help="init-method")
args = parser.parse_args()

# 初始化组中第rank个进程, icp 方法下所有 ip:port 必须与主进程保持一致 
dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.word_size)

# the sampler process, DS 将数据集划分为几个互不相交的子集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)

# ......
net = Net()
net = net.cuda()
net = torch.nn.parallel.DistributedDataParallel(net)
```

每台主机上可以开启多个进程。但是，若未为每个进程分配合适的 GPU，则同机不同进程可能会共用 GPU，应该坚决避免这种情况。

直接用 python 解释器启动各个脚本

## ENV 初始化

```python
import torch.distributed as dist
import torch.utils.data.distributed

# ......
import argparse
parser = argparse.ArgumentParser()
# 注意这个参数，必须要以这种形式指定，即使代码中不使用。因为 launch 工具默认传递该参数
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()

# ......
dist.init_process_group(backend='nccl', init_method='env://')

# ......
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)

# ......
# 根据 local_rank，配置当前进程使用的 GPU
net = Net()
device = torch.device('cuda', args.local_rank)
net = net.to(device)
net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank)
```

启动方式
```bash
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=3 --node_rank=0 --master_addr="192.168.1.201" --master_port=23456 env_init.py
```
创建 nnodes 个node, 每个 node 有 nproc_per_node 个进程(一般为 GPU 数量)，每个进程独立执行脚本训练。 node rank 确定节点的优先级, 以 0 为主节点，使用其 addr:port 作为 master 的参数 (可以用为局域网内训练), 会自动分配 node 内的各线程优先级 (local_rank)

## 可选后端
![后端](./img/backend.jpg)

## 进程间通信操作

multigpu 代表不同进程间，不同 GPU 上有 shape 相同的 Tensor, 可以通过此求和，求平均。

torch.distributed.new_group 可以将各优先级的进程组建成新组，在这些新组中进行后面的组间信息交流。返回一个 group object

# pytorch Tutorials

## Data preparation

torch.utils.data.Dataset is a abstract class , following methods should be override. `__len__` & `__getitem__` . Typically, the path and txt setup is in `__init__` and image reading is in `__getitem__`

```python
class MyDataSet(Dataset):
	def __init__(self, root_dir, csv_file, transform=None):
	"""
		the csv file contains thr images name
	""" 
	self.root_dir = root_dir
	self.transform =transform # this may a function
	self.label = pd.read_csv(csv_file) # read the label txt
	# the preprocess of data or its organization can follow
	
	def __len__(self):
		return len(self.label)
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_name = o.path.join(self.root_dir, self.label.iloc[idx, 0 ]) # column 1 is the name
		img = io.imread(img_name)
		labels = self.label[idx, 0]
		labels = np.array(labels).astype(np.float32).reshape(...) 
		sample = {'image':image, 'labels':label}
		if self.transform :
			sample = self.transform(sample)
		return sample
```

## Visualization

tensorboard
```python
from torch.utils.tensorboard import SummmaryWritter
writer = SummaryWriter('runs/test') # construct a writer

# for a given image
img_grid = torchvision.utils.make_grid(images)
writer.add_image('batch_images', img_grid) # show the images in a batch
# add network
writer.add_graph(net, input)
writer.close()

# add projector to visualize data
def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)
    perm = torch.randperm(len(data))  # shuffle the index
    return data[perm][:n], labels[perm][:n]

images, labels = select_n_random(trainset.data, trainset.targets)
class_labels = [classes[lab] for lab in labels]
features = images.view(-1, 28*28)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))

# the add_scalar can show the chne of scalar(the mean loss of 1000 batch)
```

# STN

STN 本质上是让网络习得一组变换参数。

eg. 对 2D 来说， 为[[scale_x, 0, trans_x],[0, scale_y, trans_y]], 之后根据变换关系到原 feature map 中采样

```python
# x 为输入数据
grid = F.affine_grid(theta, x.size())
x = F.grid_sample(x, grid)
```

# Jit Script

convert to torchscript and use cpp to deploy 

<https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html> (official tutorials)

# 实际编写

比如在计算$ |\vec{x_{tar}} - \vec{x_{cur}}|^{2} $ , 采用如下化简：
$$ |\vec{x_{tar}} - \vec{x_{cur}}|^{2} = \vec{x_{tar}}^{T} \vec{x_{tar}} - 2 * \vec{x_{tar}}^{T} \vec{x_{cur}} + \vec{x_{cur}}^{T} \vec{x_{cur}} $$
```python
# des_tar , des_cur 代表两帧的描述子 N*256
# terrible ways:
des_tar = torch.unsqueeze(des_tar, 0) # 1*N*256
des_cur = torch.unsqueeze(des_cur, 1) # N*1*256
compare = torch.sum((des_tar - des_cur)**2, 2) # N*N*256

# in ORB situation, the descriptor is binary 
compare = torch.sum(des_tar, 1) - 2*torch.matmul(des_tar, des_cur.t()) + torch.sum(des_cur,1)

# if descriptors are normalized : 
compare = 2 - 2*torch.matmul(des_tar, des_cur.t()) 
```

