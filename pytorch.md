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

# STN

STN 本质上是让网络习得一组变换参数。

eg. 对 2D 来说， 为[[scale_x, 0, trans_x],[0, scale_y, trans_y]], 之后根据变换关系到原 feature map 中采样

```python
# x 为输入数据
grid = F.affine_grid(theta, x.size())
x = F.grid_sample(x, grid)
```