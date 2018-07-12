import tensorflow as tf
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import cv2
import mnist_inference
import mnist_train


def imageprepare():
    """
        This function returns the pixel values.
        The input is a png file location.
    """
    file_name = '9-test.png'
    im = Image.open(file_name).convert('L')

    im.save("9-t.png")
    plt.imshow(im)
    plt.show()
    tv = list(im.getdata())

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva

result = imageprepare()
x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')

# 直接用封装好的函数计算前向传播的结果，因为测试时不关注正则化损失的值，所以为None
y = mnist_inference.inference(x, None)
saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        # 加载模型
        saver.restore(sess, ckpt.model_checkpoint_path)
        prediction = tf.argmax(y, 1)
        predint = prediction.eval(feed_dict={x: [result]}, session=sess)

        print('recognize result:')
        print(predint[0])
