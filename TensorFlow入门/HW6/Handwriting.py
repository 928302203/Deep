import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
import mnist_inference
import mnist_train

# 当鼠标按下时变为True
drawing = False
# 如果mode为true绘制矩形。按下'm' 变成绘制曲线。
#mode = True
ix, iy = -1, -1
# img = np.zeros((512, 512, 3), np.uint8)


# 创建回调函数
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    # 当按下左键是返回起始位置坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # 当鼠标左键按下并移动是绘制图形。event可以查看移动，flag查看是否按下
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing:

            # 绘制圆圈，小圆点连在一起就成了线,3代表了笔画的粗细
            cv2.circle(img, (x, y), 18, (0, 0, 0), -5)
            # 下面注释掉的代码是起始点为圆心，起点到终点为半径的
            # r=int(np.sqrt((x-ix)**2+(y-iy)**2))
            # cv2.circle(img,(x,y),r,(0,0,255),-1)
        # 当鼠标松开停止绘画。
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # if mode==True:
            # cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            # else:
            # cv2.circle(img,(x,y),5,(0,0,255),-1)


# 回调函数与OpenCV 窗口绑定在一起,
img = np.zeros((400, 400), np.uint8)
img = img + 255
cv2.namedWindow('image')
# 绑定事件
cv2.setMouseCallback('image', draw_circle)


def imageprepare():
    """
        This function returns the pixel values.
        The input is a png file location.
    """
    file_name = 'temp_image.png'
    im = Image.open(file_name).convert('L')
    im = im.resize((28, 28))
    im.save("last_image.png")
    tv = list(im.getdata())

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva


#if drawing == True:

# if k==ord('m'):
#    mode=not mode
# elif k==27:
# break

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
        while (1):
            cv2.imshow('image', img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            if k == ord('s'):
                img = np.zeros((400, 400, 3), np.uint8)
                img = img + 255
            if k == 13:
                cv2.imwrite('temp_image.png', img)
                result = imageprepare()
                predint = prediction.eval(feed_dict={x: [result]}, session=sess)

                print('recognize result:')
                print(predint[0])
        #else:


    #    time.sleep(1)
