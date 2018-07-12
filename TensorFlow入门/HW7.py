# 卷积神经网络实现卷积层和池化层
import tensorflow as tf
# [5, 5, 3, 16] 前两个代表过滤器的尺寸，第三个是当前层的深度，第四个表示过滤器的深度
filter_weights = tf.get_variable(
    'weights', [5, 5, 3, 16], initializer=tf.truncated_normal_initializer(stddev=0.1)
)
# 16表示过滤器的深度
biases = tf.get_variable(
    'biases', [16], initializer=tf.constant_initializer(0.1)
)
# 第三个参数是一个长度为4的数组，但第一位和第四位的数字要求一定是1，因为卷积层的步长只对矩阵的
# 长和宽有效，最后一个参数是填充，SAME表示添加全0填充，VALID表示不添加
conv = tf.nn.conv2d(
    input, filter_weights, strides=[1, 1, 1, 1], padding='SAME'
)
biases = tf.nn.bias_add(conv, biases)
# 将计算结果通过ReLU激活函数完成去线性化
actived_conv = tf.nn.relu(biases)
# ksize表示过滤器的尺寸,第二个参数是一个长度为4的数组，但第一位和第四位的数字要求一定是1
# 常见过滤器尺寸是[1, 3, 3, 1]和[1, 2, 2, 1]
# 还有平均池化层tf.nn.avg_pool()
pool = tf.nn.max_pool(actived_conv, ksize=[1, 3, 3, 1],
                      strides=[1, 2, 2, 1], padding='SAME')
