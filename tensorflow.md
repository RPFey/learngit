tensorflow 构建运算图，并不直接计算

eg.  y = a+b 

构建会话后才会执行计算

with tf.Session() as sess :

​	sess.run(y)