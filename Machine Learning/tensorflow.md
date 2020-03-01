# linux

当有过多提示等级，可以修改 bashrc 降低提示等级

export TF_CPP_MIN_LOG_LEVEL=2

tensorflow 构建运算图，并不直接计算，表示计算任务，图中的节点表示计算任务(operation,op)

在 session (context) 中执行图

Variable 维护状态

eg.  y = a+b   只会生成一张图，而不会得到结果

构建会话后才会执行计算，也就是操作产生的影响只会在 sess.run(op) 之后才会产生；比
较坑的地方在于只能获得最后一步的结果，看不到中间的结果。

with tf.Session() as sess :

​	sess.run(y)

或者

sess = tf.Session()

result = sess.run(y)

sess.close()

# Variable

tf.Variable() 与 tf.constant()

variable 在 会话中计算时需要初始化 

init = tf.global_variables_initializer()

with tf.Session() as sess:

​	sess.run(init)

​	....

变量 和 op 有 name 参数，可以命名

会话中赋值  tf.assign(a,b) —— 将 b 赋值给 a , 且返回是一个 op

update = tf.assign(a,b) 

sess.run(update)   才会执行赋值这个操作

## fetch&feed

sess.run([op1,op2,...])

会返回 : [result1, result2, ....],  且按顺序进行。

占位变量赋值 ： 

v=tf.placeholder(dtype= ... , shape=(...))  元组形式指明维度

维度可以用None 表示任意大小

在 sess.run(op, feed_dict={<variable_name>:  <value_>}) 执行赋值并计算。 可以直接用 ndarray 。

常用方法： 最开始用 placeholder 设置输入数据格式， 最后用 sess.run() 放入数据


# 数据读取

## batch & mini_batch

```python
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(batch_size) :
    batch_x = ...
    batch_y = ...
    sess.run(train_step, freed_dict={... : batch_x, ...:batch_ys})
```

## 读取机制

tensorflow 将要计算的数据读入一个内存队列中，与计算的线程分隔开。

文件名队列与内存队列：文件名队列是读入数据的顺序，用来表示训练的 epoch, 内存队列则是要读取的单张图

```python

#构建工作队列
input_queue = tf.train.string_input_produce([image_text.txt], num_epoches=epoch, shuffle=True)
# input 返回对应的文本文件对象，用LineReader读取
line_reader = tf.TextLineReader()
_, line = line_reader.read(input_queue)
split_line = tf.string_split([line]).values 
# 将列表中的字符串划分，返回对象包括(indices, values, dense_shape) , values 是具体的字符串，indice 指标(i,j) 第i个字符串的第j 个词。
left_image_path = tf.string_join([self.data_path, split_line[0]])
# string_join 是路径相连 相当于 os.path.join

tf.image.decode_png(tf.read_file(image_path)) # 读取
orig_height = tf.cast(tf.shape(image)[0], "float32") # 获取大小
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

optimizer 
```python
optimizer=tf.train.GradientDescent(0.2) --> learning rate
train = optimizer.minimize(loss)
with tf.Session() as sess :
​	sess.run(train)
```

logistic regression

least-square :

``` python
loss  = tf.reduce_mean(tf.square(y_data-y))
```

主要的思想是先初始化变量（Variable）, 构建计算题（计算误差，训练）

但是这里好像要自己构建 各个层的形状，参数。

在会话中运行。

softmax regression & cross entropy

```python
y = tf.nn.softmax(x) 
```
cross entropy :

$H_{y'}(y) = -\sum(y_{i}'log(y_i))$

$y_{i}' 为实际概率值而y_{i}为预测值$
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
with slim.arg_scope([slim.con2d], 
                    weights_regularizer = slim.l2_regularizer(0.0004),
                    activation_function = leaky_relu,
                    reuse = True) # 等
    conv1 = slim.conv2d(image, 16, [3,3], stride=2, scope="conv1")
    conv2 = slim.conv2d(conv1, 16, [3,3], stride=1, scope="conv2")
    # ...
# 所有层都会具有 leaky_relu , l2_regularizer
```

# operation

matmul(a,b) 矩阵乘法

tf.equal 判断元素是否相等， 返回布尔类型tensor

tf.cast(.. , dtype) 转变类型

tf.reduce_mean(..) 获得平均值

tf.split(value, num_or_size_split, axis, num=None, name='split') : 将 Tensor 在指定维度上划分

tf.cond(pred, fn1, fn2) <==> if pred, do fn1; else fn2

tf.set_shape()  相当于 reshape ; tf.stack([...]) 合并 ； tf.matrix_inverse 逆矩阵 ； 

tf.pad(tensor, padding)  在 tensor 上打补丁, padding=[[d1, d2],[d11, d22],[d21, d22]] 分别对应各个维度前后打多少。

# Visualization

name_scope & variable_scope

name_scope() 与 Variable() 使用，是为了管理变量的命名空间提出，在 tensorboard 中显示。

variable_scope() 与 get_variable() 使用，实现变量的共享。get_variable() 搜索相应的变量名称，如果没有会新建，有则提取同样名字的变量。

get_trainable() 获得所有训练参数列表。get_collection(tf.Graphkeys.TRAIANABLE_VARIABLES, scope="...") scope 可以使用正则表达式。

两者会开辟不同的空间。name_scope, 与 variable_scope 开辟的域会加在 Variable 声明的变量上； 而只有 variable_scope 声明的域会加在 get_variable() 创建的变量上。

关于 reuse， 在构建过程中如果同时有多个 example 经过同一个神经网络，就需要用到 reuse; 方法是将所经过的层设置为相同的名称，并将参数 reuse 设置为 true / 或者是 get_variable() 针对单个 tensor。 比如：

1. 在 slim 中，在最开时的slim.arg_scope() 设置 reuse=True ; 
```python
feature1 = feature_pyramid_flow(image1, reuse=False)
feature2 = feature_pyramid_flow(image2, reuse=True)
# feature 经过相同的网络， 在函数中设置为 reuse
```