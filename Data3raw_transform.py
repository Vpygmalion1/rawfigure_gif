import numpy as np
import os
import cv2
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import random


# step1：读取源文件，查看维度
c = 230
w = 512
h = 512
path = r'./data_set/Data3_512_512_230_16.raw'
size = (c,w,h)

rawData = np.fromfile(path, dtype='uint16')
print(np.shape(rawData))  # 63963136

# # step2：转换源文件维度
reshapeRawData = np.reshape(rawData, size)


# 创建文件夹
# os.makedirs('data_set/Data3')
# os.makedirs('data_set/Data3_noise')
# os.makedirs('./data_set/Data3_concat')

def make_feature():
    for i in range(len(reshapeRawData)):
        im1 = Image.fromarray(reshapeRawData[i])
        im1.save(f'./data_set/Data3/{i}.png')

# 制作gif
def create_gif(image_list, gif_name, duration = 1.0):
    '''
    :param image_list: 这个列表用于存放生成动图的图片
    :param gif_name: 字符串，所生成gif文件名，带.gif后缀
    :param duration: 图像间隔时间
    :return:
    '''
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))

    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def gif_main(path,name):
    #这里放上自己所需要合成的图片
    fileList = os.listdir(path)
    fileList.sort(key=lambda x:int(x[:-4]))
    print(fileList)
    images_path=[]
    for i in fileList:
        firstImgPath = os.path.join(path, i)
        images_path.append(firstImgPath)
    print(images_path)
    gif_name = name
    duration = 0.0005
    create_gif(images_path, gif_name, duration)
    return images_path


def sp_noise(image,prob):
    '''
    添加椒盐噪声
    image:原始图片
    prob:噪声比例
    '''
    image = cv2.imread(image)
    output = np.zeros(image.shape,np.uint8)
    noise_out = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()#随机生成0-1之间的数字
            if rdn < prob:#如果生成的随机数小于噪声比例则将该像素点添加黑点，即椒噪声
                output[i][j] = 0
                noise_out[i][j] = 0
            elif rdn > thres:#如果生成的随机数大于（1-噪声比例）则将该像素点添加白点，即盐噪声
                output[i][j] = 255
                noise_out[i][j] = 255
            else:
                output[i][j] = image[i][j]#其他情况像素点不变
                noise_out[i][j] = 100
    # result = [noise_out,output]#返回椒盐噪声和加噪图像
    return output

def noise_data(images):
    for i in range(c):
        # print(image)
        output = sp_noise(images[i],0.1)
        plt.imshow(output)
        output = Image.fromarray(output)
        output.save(f'./data_set/Data3_noise/{i}.png')

def figure_concat():
    for i in range(c):
        im1 = cv2.imread(f'./data_set/Data3_noise/{i}.png')
        im2 = cv2.imread(f'./data_set/Data3/{i}.png')
        fig = plt.figure(1)
        plt.subplot(1,2,1)
        plt.imshow(im1)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1,2,2)
        plt.imshow(im2)
        plt.xticks([])
        plt.yticks([])
        plt.draw()
        plt.savefig(f'./data_set/Data3_concat/{i}.png')
        plt.pause(0.1)  # 间隔的秒数：0.1s
        plt.close(fig)

if __name__ == '__main__':
    make_feature()
    images_gif = gif_main(r'./data_set/Data3','data3.gif')
    noise_data(images_gif)
    noise_gif = gif_main(r'./data_set/Data3_noise','data3_noise.gif')
    figure_concat()
    concat_gif = gif_main(r'./data_set/Data3_concat','data3_concat.gif')

