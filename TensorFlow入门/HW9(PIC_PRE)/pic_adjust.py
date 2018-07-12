import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

image_raw_data = tf.gfile.FastGFile('cat.jpg', "rb").read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)

    # # 1. 改变大小
    # resized = tf.image.resize_images(img_data, [300, 300], method=0)
    # print(resized.get_shape())
    # # 注：resized.eval()中要加上session=sess
    # resized = np.asarray(resized.eval(session=sess), dtype='uint8')
    # plt.imshow(resized)
    # plt.show()

    # # 2. 自动裁剪和自动填充
    # croped = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
    # padded = tf.image.resize_image_with_crop_or_pad(img_data, 3000, 3000)
    # croped = np.asarray(croped.eval(), dtype='uint8')
    # padded = np.asarray(padded.eval(), dtype='uint8')
    # plt.imshow(croped)
    # plt.show()
    # plt.imshow(padded)
    # plt.show()

    # # 截取中间50%的图像
    # central_cropped = tf.image.central_crop(img_data, 0.5)
    # central_cropped = np.asarray(central_cropped.eval(session=sess),
    #                              dtype='uint8')
    # plt.imshow(central_cropped)
    # plt.show()

    # # 3.图像翻转(上下、左右、沿对角线旋转)
    # flipped1 = tf.image.flip_up_down(img_data)
    # flipped2 = tf.image.flip_left_right(img_data)
    # transposed = tf.image.transpose_image(img_data)
    #
    # # 以随机概率进行翻转，可以零成本增加样本的多样性
    # flipped3 = tf.image.random_flip_left_right(img_data)
    # flipped4 = tf.image.random_flip_up_down(img_data)

    # 4.亮度、对比度、饱和度、色相等处理

    # # 5.处理标注框
    # # tf.image.draw_bounding_boxes函数要求图像矩阵中的数字为实数
    # # tf.image.draw_bounding_boxes函数图像输入的是一个batch的数据，也就是多张
    # # 图像组成的四维矩阵，座椅要加一维
    # img_data = tf.image.resize_images(img_data, [180, 267], method=1)
    # batched = tf.expand_dims(
    #     tf.image.convert_image_dtype(img_data, tf.float32), 0
    # )
    # # [y_min, x_min, y_max, x_max]
    # # 这是图像的相对位置，例如在[180， 267]的图像中，[0.35,0.47,0.5,0.56]代表
    # # 了从(63, 125)到(90, 150)的图像
    # boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    # result = tf.image.draw_bounding_boxes(batched, boxes)
    # # result = tf.cast(result, dtype=tf.uint8)
    # # print(result[0].eval())
    # plt.imshow(result[0].eval())
    # plt.show()

    # 6.随机切片
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    # 通过标注框的方式告诉随机截取图像的算法哪些部分是有“信息量”的
    # f.image.sample_distorted_bounding_box函数为图像生成单个随机变形的边界框。
    # 函数输出的是可用于裁剪原始图像的单个边框。返回值为3个张量：begin，size和 bboxes。
    # 前2个张量用于 tf.slice 剪裁图像。后者可以用于 tf.image.draw_bounding_boxes 函数来画出边界框
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img_data), bounding_boxes=boxes
    )
    batched = tf.expand_dims(
            tf.image.convert_image_dtype(img_data, tf.float32), 0
    )
    image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)
    disorted_image = tf.slice(img_data, begin, size)
    plt.imshow(disorted_image.eval())
    plt.show()
