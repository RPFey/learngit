# linux

当有过多提示等级，可以修改 bashrc 降低提示等级

export TF_CPP_MIN_LOG_LEVEL=2

# 变量

## tf.Variable()

variable 在 会话中计算时需要初始化 

```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
	....
    
# 如果想要获取变量的值
weight = tf.Variable([0.]*num_coff, name="parameters")
# ... (run trainig op)
w = sess.run(weight) # ndarray type
```

变量 和 op 有 name 参数，可以命名

会话中赋值  tf.assign(a,b) —— 将 b 赋值给 a , 且返回是一个 op eg.

```python
self.epoch_add_op = self.epoch.assign(self.epoch + 1) # epoch 自加操作
```

如果要查看全局的变量（以检查变量是否重复使用）

```python
print([n.name for n in tf.get_default_graph().as_graph_def().node if 'Variable' in n.op])

# or
print(tf.global_variables())
```



## gradient

如果对梯度有什么操作的话，可以构建如下图:

```python
self.params = tf.trainable_variables() # get all the trainable gradients
gradients = tf.gradients(self.loss, self.params)  # 相当于是求梯度的操作
clipped_gradients, gradient_norm = tf.clip_by_global_norm(
	gradients, max_gradient_norm)
```

这里是对所有参数的梯度进行一次放缩到 max_gradient_norm 。详细可见 [Pascau et al., 2012](http://arvix.org/abs/1211.5063.pdf)



## fetch & feed

sess.run([op1,op2,...], {...})

会返回 : [result1, result2, ....],  且按顺序进行。

占位变量赋值 ： 

v=tf.placeholder(dtype= ... , shape=(...))  元组形式指明维度，维度可以用None 表示任意大小

在 sess.run(op, feed_dict={<variable_name>:  <value_>}) 执行赋值并计算。 可以直接用 ndarray 。如果有多个操作，各个 placeholder 只需要赋值一次即可。

常用方法： 最开始用 placeholder 设置输入数据格式， 最后用 sess.run() 放入数据


# 数据读取

## batch & mini_batch

```python
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(batch_size) :
    batch_x = ...
    batch_y = ...
    sess.run(train_step, feed_dict={... : batch_x, ...:batch_ys})
```

这个算是比较常见的，从 Dataset 类中提取数据，然后做成 feed_dict 喂入。

## 读取机制

tensorflow 将要计算的数据读入一个内存队列中，与计算的线程分隔开。

文件名队列与内存队列：文件名队列是读入数据的顺序，用来表示训练的 epoch, 内存队列则是要读取的单张图

```python
# 如果是要从文件夹下读取
filenames = tf.train.match_filenames_once('./data/*.txt')
count_num_files = tf.size(filenames)
filename_queue = tf.train.string_input_producer(filenames)
reader = tf.WholeFileReader()
filename, file_contents = reader.read(filename_queue)


#构建工作队列
input_queue = tf.train.string_input_produce([image_text.txt], num_epoches=epoch, shuffle=True)
# input 返回对应的文本文件对象，用LineReader读取
line_reader = tf.TextLineReader()
_, line = line_reader.read(input_queue)
split_line = tf.string_split([line]).values 
# 将列表中的字符串划分，返回对象包括(indices, values, dense_shape) , values 是具体的字符串，indice 指标(i,j) 第i个字符串的第 j 个词。
left_image_path = tf.string_join([self.data_path, split_line[0]])
# string_join 是路径相连 相当于 os.path.join

tf.image.decode_png(tf.read_file(image_path)) # 读取
origin_height = tf.cast(tf.shape(image)[0], "float32") # 获取大小
image = tf.image.convert_image_dtype(image, tf.float32) # 转换类型
image = tf.image.resize_images(
            image, [self.opt.img_height, self.opt.img_width],
            tf.image.ResizeMethod.AREA) # 缩放

self.data_batch = tf.train.shuffle_batch([
            left_image, right_image, next_left_image, next_right_image,
            proj_cam2pix, proj_pix2cam
        ], opt.batch_size, capacity, min_after_dequeue, 10)
# tf.train.shuffle_batch([tensor_list], batch_size, capacity, min_after_dequeue, workers)
# 从工作队列里面提取，组成一个训练样本(由tensor_list 表示), min_after_queue 指抽取完后队列里面应该保留的数量，保证能够打乱。
```

# loss function

## optimizer

```python
optimizer=tf.train.GradientDescent(0.2) # --> learning rate
train = optimizer.minimize(loss)
with tf.Session() as sess :
    sess.run(train)
```

当然了，如果想要指定步长，在不同区间里面有不同的 learning rate,

```python
boundaries = [80, 120]
values = [ self.learning_rate,  self.learning_rate*0.1, self.learning_rate*0.01]
# values 中的值可以是浮点数或者 Variable
lr = tf.train.piecewise_constant(self.epoch, boundaries, values) # epoch 必须是 Variable
self.opt = tf.train.AdamOptimizer(lr) # optimzer
```

## least-square

``` python
loss  = tf.reduce_mean(tf.square(y_data-y))
```

主要的思想是先初始化变量（Variable）, 构建计算题（计算误差，训练）

但是这里好像要自己构建 各个层的形状，参数。

在会话中运行。

## softmax regression & cross entropy

```python
y = tf.nn.softmax(x)
```

cross entropy :
$$
H_{y'}(y) = -\sum(y_{i}'log(y_i))
$$

$$
y_{i}' 为实际概率值而y_{i}为预测值
$$

```python
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
```

# Module 

slim module :
tf.contrib.slim

```python
net = slim.conv2d(input, 32, [3,3])
# 直接定义卷积层

# 采用 arg_scope() 批量定义：
with slim.arg_scope([slim.conv2d], # 列表中是指调用哪些层
                    weights_regularizer = slim.l2_regularizer(0.0004),
                    activation_function = leaky_relu,
                    reuse = True) # 等
    conv1 = slim.conv2d(image, 16, [3,3], stride=2, scope="conv1")
    conv2 = slim.conv2d(conv1, 16, [3,3], stride=1, scope="conv2")
    # ...
# 所有层都会具有 leaky_relu , l2_regularizer
```

## save and load model

```python
class Model(object):
    def __init__(self):
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.v2,
                                   max_to_keep=10, pad_step_number=True,
                                    keep_checkpoint_every_n_hours=1.0)

model = Model()
model.saver.restore(sess, tf.train.lastest_checkpoint(save_model_dir)) # load model

model.saver.save(sess, os.path.join(save_model_dir, 'checkpoint'), global_step=model.global_step) # save model
```

这些都应该是在 sess 建立好之后。

# operation

matmul(a,b) 矩阵乘法

tf.equal 判断元素是否相等， 返回布尔类型tensor; 相对的是 tf.not_equal . 这两个操作常用来作为 mask

```python
mask = tf.not_equal(tf.reduce_max(
        self.features, axis=2, keep_dims=True), 0)
```

tf.cast(.. , dtype) 转变类型

tf.reduce_mean(..) 获得平均值 / tf.reduce_max 获得最大值。

tf.split(value, num_or_size_split, axis, num=None, name='split') : 将 Tensor 在指定维度上划分

tf.cond(pred, fn1, fn2) \<==\> if pred, do fn1; else fn2

tf.set_shape()  相当于 reshape ; tf.stack([...]) 合并 ； tf.matrix_inverse 逆矩阵 ；

tf.pad(tensor, padding)  在 tensor 上打补丁, padding=[[d1, d2],[d11, d22],[d21, d22]] 分别对应各个维度前后打多少。

tf.slice(input, begin, size, name=None) 在 input 张量上截取。 begin[i] 代表第 i 个唯独上的 offset, size[i] 代表第 i 个维度上截取的数量

# Visualization

## name_scope & variable_scope

name_scope() 与 Variable() 使用，是为了管理变量的命名空间提出，在 tensorboard 中显示。

variable_scope() 与 get_variable() 使用，实现变量的共享，即重复使用同一张网络。get_variable() 搜索相应的变量名称，如果没有会新建，有则提取同样名字的变量。

```python
def get_scope_variable(scope, var, shape=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        v = tf.get_variable(var, shape)
    return v

v1 = get_scope_variable("foo", "v", [1])
v2 = get_scope_variable("foo", 'v') # reuse varibale named 'v' under 'foo' scope
```



get_trainable() 获得所有训练参数列表。get_collection(tf.Graphkeys.TRAIANABLE_VARIABLES, scope="...") scope 可以使用正则表达式。

两者会开辟不同的空间。name_scope, 与 variable_scope 开辟的域会加在 Variable 声明的变量上； 而只有 variable_scope 声明的域会加在 get_variable() 创建的变量上。

关于 reuse， 在构建过程中如果同时有多个 example 经过同一个神经网络，就需要用到 reuse; 方法是将所经过的层设置为相同的名称，并将参数 reuse 设置为 true / 或者是 get_variable() 针对单个 tensor。 比如：

1. 在 slim 中，在最开时的slim.arg_scope() 设置 reuse=True ; 
```python
feature1 = feature_pyramid_flow(image1, reuse=False)
feature2 = feature_pyramid_flow(image2, reuse=True)
# feature 经过相同的网络， 在函数中设置为 reuse
```

2. VoxelNet 中设计 VEF 层

```python
class FeatureNet(object):
    def __init__(self):
        # other parameters here
        self.features = tf.place_holder(tf.float32, [None, cfg.VOXEL_COUNT, 7], name='features') # 输入是 N * T * 7
        
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.vfe1 = VEFLayer(32, 'VFE-1')
            self.vfe2 = VEFLayer(128, 'VFE-2') # 这两步相当于构建 VFE 层图
        mask = tf.not_equal(tf.reduce_max(
        	self.features, axis=2, keep_dims=True), 0) 
        # self.features 由之前的place_holder 表示
        x = self.vfe1.apply(self.features, mask, self.training) 
        x = self.vfe2.apply(x, mask, self.training) # 构建输入, N * T * 128
        
        voxelwise = tf.reduce_max(x, axis=1) # N * 128
        self.outputs = tf.scatter_nd(
        	self.coordinate, voxelwise, [self.batch_size, 10, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128]) # 这里 outputs 是构建的运算，外部用 feature.outputs　获取。
        
        
class VFElayer(object):
    def __init__(self, out_channel, name):
        super(VFELayer,self).__init__()
        self.units = int(out_channel / 2)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope: 
            # 这里就是在上一级上加一个'VFE-1/'目录
            self.dense = tf.layers.Dense(self.units, tf.nn.relu, 
                                         name='dense', _reuse=tf.AUTO_REUSE, _scope=scope)
            self.batch_norm = tf.layers.BatchNormalization(
            	name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)
            # 这两个相当于 '/VFE-1/dense' 与 'VFE-1/batch_norm'
            
    def apply(self, inputs, mask, training):
        pointwise = self.batch_norm.apply(self.apply(inputs), training)
        # tf layer 用 apply 当作 forward
        aggregated = tf.reduce_max(pointwise, axis=1, keep_dims=True)
        repeated = tf.tile(aggregated, [1, cfg.VOXEL_COUNT, 1])
        concatenate = tf.concat([pointwise, repeated], axis=2)
        mask = tf.tile(mask, [1, 1, 2*self.units])
        return tf.multiply(concatenated, tf.cast(mask, tf.float32))
    
```

   

## tensorboard



# Multi-GPU & distributed training

```python
gpu_options = tf.GPUOptions(pre_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION, # 1
                           visible_device_list=cfg.GPU_AVAILABLE,　# '0,1'
                           allow_growth=True)

config = tf.ConfigProto(gpu_options=gpu_options,
                       device_count={
                           "GPU":cfg.GPU_USE_COUNT, # number of GPUs
                       },
                       allow_soft_placement=True)

with tf.Session(config=config) as sess:
    # training procedure
```

GPUOptions :

* pre_process_gpu_memory_fraction : 每块GPU 使用显存上限的百分比。
* visible_device_list : 使用 GPU 的 ID 号
* allow_growth : 分配器将不会指定所有的GPU内存而是根据需求增长，但是由于不会释放内存，所以会导致碎片

ConfigProto : 

* log_device_placement=True ： 是否打印设备分配日志

* allow_soft_placement=True ： 如果你指定的设备不存在，允许TF自动分配设备

* inter_op_parallelism_threads=0：设置线程一个操作内部并行运算的线程数，如果设置为０，则表示以最优的线程数处理

* intra_op_parallelism_threads=0：设置多个操作并行运算的线程数

在具体训练时，需要在各GPU 上分别建立网络训练，然后在各GPU之间平均梯度。（具体见 UndepthFlow）
