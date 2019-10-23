# linux

当有过多提示等级，可以修改 bashrc 降低提示等级

export TF_CPP_MIN_LOG_LEVEL=2



tensorflow 构建运算图，并不直接计算，表示计算任务，图中的节点表示计算任务(operation,op)

在 session (context) 中执行图

Variable 维护状态



eg.  y = a+b   只会生成一张图，而不会得到结果

构建会话后才会执行计算，也就是操作产生的影响只会在 sess.run(op) 之后才会产生

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

会话中赋值  tf.assign(a,b ) —— 将 b 赋值给 a , 且返回是一个 op

update = tf.assign(a,b) 

sess.run(update)   才会执行赋值这个操作

## fetch&feed

sess.run([op1,op2,...])

会返回 : [result1, result2, ....],  且按顺序进行。

占位变量赋值 ： 

v=tf.placeholder(dtype= ... , shape=(...))  元组形式指明维度

维度可以用None 表示任意大小

在 sess.run(op, feed_dict={<variable_name>:  <value_>}) 执行赋值并计算。 可以直接用 ndarray 。

# logistic regression

least-square :

loss  = tf.reduce_mean(tf.square(y_data-y))

optimizer=tf.train.GradientDescent(0.2) --> learning rate

train = optimizer.minimize(loss)

with tf.Session() as sess :

​	sess.run(train)

主要的思想是先初始化变量（Variable）, 构建计算题（计算误差，训练）

但是这里好像要自己构建 各个层的形状，参数。

在会话中运行。

# operation

## matrix

matmul(a,b) 矩阵乘法

