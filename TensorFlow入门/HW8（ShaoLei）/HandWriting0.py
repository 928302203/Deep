import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
import mnist_inference
import mnist_train


def imageprepare():
    """
        This function returns the pixel values.
        The input is a png file location.
    """
    file_name = 'temp_image.png'
    im = Image.open(file_name).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    '''
    tv = list(im.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    tva = np.reshape(tva, (28, 28))
    '''
    im_arr = np.array(im)
    num0 = 0
    num255 = 0
    threshold = 100
    for x in range(28):
        for y in range(28):
            if im_arr[x][y] > threshold:
                num255 = num255 + 1
            else:
                num0 = num0 + 1
    if (num255 > num0):
        # print("convert!")
        for x in range(28):
            for y in range(28):
                if (im_arr[x][y] < threshold):  im_arr[x][y] = 0
    tva = np.uint8(im_arr)
    out = Image.fromarray(tva)
    out.save('3.png')
    # im = Image.open(file_name)
    # im.save("last_image.jpg")

    return tva


#if drawing == True:

# if k==ord('m'):
#    mode=not mode
# elif k==27:
# break
#
x = tf.placeholder(
        tf.float32, [1, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE,
                     mnist_inference.NUM_CHANNELS], name='x-input')

# 直接用封装好的函数计算前向传播的结果，因为测试时不关注正则化损失的值，所以为None
y = mnist_inference.inference(x, False, None)
saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        # 加载模型
        saver.restore(sess, ckpt.model_checkpoint_path)
        prediction = tf.argmax(y, 1)
        result = imageprepare()
        result = result[np.newaxis, :, :, np.newaxis]
        predint = prediction.eval(feed_dict={x: result}, session=sess)
        print('recognize result:')
        print(predint[0])
        #else:


    #    time.sleep(1)
