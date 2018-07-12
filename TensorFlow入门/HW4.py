# 滑动平均模型

import tensorflow as tf

# 定义一个变量用于计算滑动平均
v1 = tf.Variable(0, dtype=tf.float32)
# 定义step变量模拟神经网络中迭代的次数
step = tf.Variable(0, trainable=False)
# 定义一个滑动平均的类 初始化给定衰减率0.99和控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99, step)

maintain_average_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    print(sess.run([v1, ema.average(v1)]))
    # 更新v1的值为5
    sess.run(tf.assign(v1, 5))
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(step, 10000))
    sess.run(tf.assign(v1, 10))
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))
