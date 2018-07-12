import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("../MNIST_DATA/", one_hot=True)

# MNIST数据集相关的参数
INPUT_NODE = 784  # 输入层的节点数
OUTPUT_NODE = 10  # 输出层的节点数

# 配置神经网络的参数
LAYER1_NODE = 500  # 隐含层的节点数
BATCH_SIZE = 100  # 一个batch钟训练数据的个数

LEARNING_RATE_BASE = 0.8  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEP = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率

# 一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果
def inference(input_tensor, avg_class, weights1, biases1,
              weights2, biases2):
    if avg_class ==None:
        # 计算隐含层的前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

    else:
        # 首先使用avg_class.average函数计算得出变量的滑动平均值
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) +
                            avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + \
               avg_class.average(biases2)

def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    y = inference(x, None, weights1, biases1,weights2, biases2)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )

    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # average_y是一个batch_size*10的二维数组，每一行表示一个样例的前向传播过程
    # inference（）描述了通过网络的正向传递
    average_y = inference(x, variable_averages, weights1, biases1, weights2,
                          biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y,
                                                                   tf.arg_max(y_, 1))
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失
    regularization = regularizer(weights1) +regularizer(weights2)
    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,  # 当前迭代次数
        mnist.train.num_examples / BATCH_SIZE,  # 过完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY
    )

    # 使用tf.train.GradientDescentOptimizer优化算法优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss, global_step=global_step
        )

    # 在训练神经网络模型时，每过一遍数据既要反向传播来更新神经网络中的参数，
    # 又要更新每一个参数的滑动平均值，为了完成多个操作，Tensorflow提供了
    # tf.control_dependencies和tf.group两种机制，下面两行程序和
    # train_op = tf.group(train_step, variables_averages_op)是等价的
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # tf.argmax是用于得到正确答案对应的类别编号，其中第二个参数“1”表示选取最
    # 大值的操作仅在第一个维度中进行，也就是说，只在每一行选取最大值对应的下标
    # 于是得到的结果是一个长度为batch的一维数组，这个一位数组中的值就表示了每
    # 一个样例对应的数字识别结果。tf.equal判断两个张量的每一维是否相等，相等返回True
    correct_prediction = tf.equal(tf.arg_max(average_y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 声明tf.train.Saver()类用于保存模型
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}
        test_feed = {x:mnist.test.images, y_:mnist.test.labels}

        for i in range(TRAINING_STEP):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy "
                      "using average model is %g " % (i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x:xs, y_:ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy "
              "using average model is %g " % (i, test_acc))

        saver.save(sess, './HW5_Saver/model.ckpt')

def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST_DATA/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
