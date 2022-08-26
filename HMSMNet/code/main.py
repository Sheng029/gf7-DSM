from hmsmnet import HMSMNet
from disparity import genDisparityMap, writeTif
import time
import tensorflow as tf


def set_gpu(memory):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*memory)])


def train_only(data_dir, weights_save_path, epochs, batch_size, pre_trained_weights, min_disp, max_disp):
    '''
    训练模型，不验证
    :param data_dir:
    :param weights_save_path:
    :param epochs:
    :param batch_size:
    :param pre_trained_weights:
    :return:
    '''
    net = HMSMNet(1024, 1024, 1, min_disp, max_disp)
    net.build_model()
    net.train_only(data_dir, weights_save_path, epochs, batch_size, pre_trained_weights)


def weights2pb(weights, pb_dir, height, width, min_disp, max_disp):
    '''
    将.h5权重文件转换成.pb，pb既可用于C++也可用于Python
    :param weights: 训练好的权重文件
    :param pb_dir: pb文件存放的文件夹（不必手动创建）
    :return:
    '''
    net = HMSMNet(height, width, 1, min_disp, max_disp)   # 参数1,2,4,5可以自行指定，如(4096, 4096, 1, -192.0, 128.0)，但必须是64的倍数
    net.build_single_output_model()
    net.model.load_weights(weights, by_name=True)
    net.model.save(filepath=pb_dir, include_optimizer=False)


def predict_whole(left_rs_img_path, right_rs_img_path, height, width, pb_dir, disp_path):
    '''
    直接预测一整景影像的视差图
    :param left_rs_img_path: 左核线影像路径
    :param right_rs_img_path: 右核线影像路径
    :param height: 分块高度，构建pb时是多大就是多大
    :param width: 分块宽度，构建pb时是多大就是多大
    :param pb_dir: pb的文件夹
    :param disp_path: 视差图保存路径
    :return:
    '''
    t1 = time.time()
    disp = genDisparityMap(left_rs_img_path, right_rs_img_path, height, width, pb_dir)
    writeTif(disp, disp_path)
    t2 = time.time()
    print(t2 - t1)


if __name__ == '__main__':
    set_gpu(24)

    # data_dir = '/home/hesheng/Dataset/MIX'
    # weights_save_path = '/home/hesheng/Networks/HMSMNet/weights/HMSMNetMIX{epoch:003d}.h5'
    # epochs = 20
    # batch_size = 2
    # pre_trained_weights = '/home/hesheng/Networks/HMSMNet/weights/HMSMNetWHU.h5'
    # min_disp, max_disp = -128.0, 64.0
    # train_only(data_dir, weights_save_path, epochs, batch_size, pre_trained_weights, min_disp, max_disp)

    weights = '/home/hesheng/Networks/HMSMNet/weights/HMSMNetMIX015.h5'
    pb_dir = '/home/hesheng/Networks/HMSMNet/model/HMSMNetMIX2048P'
    height, width = 2048, 2048
    min_disp, max_disp = -192.0, 128.0
    weights2pb(weights, pb_dir, height, width, min_disp, max_disp)
