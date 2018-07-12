import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
import mnist_inference
import mnist_train

# 非常注意的一点是：！！！
# 对图片进行数组描述时，Height用y表示，Width用x表示
# 当鼠标按下时变为True
drawing = False
# 如果mode为true绘制矩形。按下'm' 变成绘制曲线。
# mode = True
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
            cv2.circle(img, (x, y), 10, (0, 0, 0), -5)
            # 绘制圆圈，小圆点连在一起就成了线,3代表了笔画的粗细
            # cv2.circle(img, (x, y), 8, (0, 0, 0), -5)
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
img = np.zeros((280, 280), np.uint8)
img = img + 255
cv2.namedWindow('image')
# 绑定事件
cv2.setMouseCallback('image', draw_circle)


# 找到图片有颜色区块
def find_min_max(fg):
    x_min = 100
    x_max = -1
    y_min = 100
    y_max = -1
    fgH, fgW = fg.shape[:2]
    for x in range(fgW):
        for y in range(fgH):
            if fg[y, x] == 0:
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x

    return x_min, x_max, y_min, y_max


# 两个不同大小的图片合并
def mergeImage(bg, fg, x, y):
    bgH, bgW = bg.shape[:2]
    fgH, fgW = fg.shape[:2]

    x_min, x_max, y_min, y_max = find_min_max(fg)
    print(x_min,x_max,y_min,y_max)
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            if (i-x<bgW and j-y<bgH and j-y>0 and i-x>0):
                bg[j-y, i-x] = fg[j, i]

    return bg


# 创建新的黑色图片
def createBianryImage(bg=(0, 0, 0), width=28, height=28):
    channels = 1

    image = np.zeros((width, height, channels), np.uint8) + 255  # 生成一个空灰度图像
    # cv2.rectangle(image,(0,0),(width,height),bg,1, -1)

    return image.reshape(width, height)


# 求像素重心。传入二值图像，其中白色点算重量，黑色点为空
def getBarycentre(image):
    h, w = image.shape[:2]

    sumWeightW = 0
    sumWeightH = 0

    count = 0

    for i in range(h):
        for j in range(w):
            if (image[i, j] < 200):
                sumWeightH += j
                sumWeightW += i
                count += 1

    if (count == 0):
        count = 1
    return (round(sumWeightW / count), round(sumWeightH / count))


def imageprepare():
    """
        This function returns the pixel values.
        The input is a png file location.
    """
    file_name = 'temp_image.png'
    im = Image.open(file_name).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im.save("last_image1.png")

    # tv = list(im.getdata())  # get pixel values
    # # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    # tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    # tva = np.reshape(tva, (28, 28))
    im_arr = np.array(im)
    num0 = 0
    num255 = 0
    threshold = 230
    width = im_arr.shape[0]
    height = im_arr.shape[1]
    for w in range(0, width):
        for h in range(0, height):
            if im_arr[w][h] > threshold:
                num255 = num255 + 1
            else:
                num0 = num0 + 1
    if (num255 > num0):
        # print("convert!")
        for w in range(0, width):
            for h in range(0, height):
                if (im_arr[w][h] < threshold):  im_arr[w][h] = 0

    # 求像素重心
    bcH, bcW = getBarycentre(im_arr)
    print(bcW, bcH)
    # 叠加到28x28的黑色图片上
    xOffset = round(bcW - 28 / 2)
    yOffset = round(bcH - 28 / 2)
    print(xOffset, yOffset)
    im_arr = mergeImage(createBianryImage(), im_arr, xOffset, yOffset)

    for w in range(0, width):
        for h in range(0, height):
            im_arr[w][h] = 255 - im_arr[w][h]

    tva = np.uint8(im_arr)
    out = Image.fromarray(tva)
    out.save('last_image2.png')
    return tva


#if drawing == True:

# if k==ord('m'):
#    mode=not mode
# elif k==27:
# break

x = tf.placeholder(
        tf.float32, [1, mnist_inference.IMAGE_SIZE,
                     mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS], name='x-input')

# 直接用封装好的函数计算前向传播的结果，因为测试时不关注正则化损失的值，所以为None
y = mnist_inference.inference(x, False, None)
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
                img = np.zeros((280, 280), np.uint8)
                img = img + 255
            if k == 13:
                cv2.imwrite('temp_image.png', img)
                result = imageprepare()
                result = result[np.newaxis, :, :, np.newaxis]
                predint = prediction.eval(feed_dict={x: result}, session=sess)
                print('recognize result:')
                print(predint[0])
        #else:


    #    time.sleep(1)

