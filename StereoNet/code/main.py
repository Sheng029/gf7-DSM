from stereonet import StereoNet
from disparity import genDisparityMap, writeTif
import time
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


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
    net = StereoNet(1024, 1024, 1, min_disp, max_disp)
    net.build_model()
    net.train_only(data_dir, weights_save_path, epochs, batch_size, pre_trained_weights)


def weights2pb(weights, pb_dir, height, width, min_disp, max_disp):
    '''
    将.h5权重文件转换成.pb，pb既可用于C++也可用于Python
    :param weights: 训练好的权重文件
    :param pb_dir: pb文件存放的文件夹（不必手动创建）
    :return:
    '''
    net = StereoNet(height, width, 1, min_disp, max_disp)   # 参数1,2,4,5可以自行指定，如(4096, 4096, 1, -192.0, 128.0)，但必须是64的倍数
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


def convert_h5to_pb_two_input(model, pb_path):
    '''
    支持模型有两个输入
    Param model: 加载h5之后的模型
    Param pb_path: 输出pb的路径
    '''
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
    [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype),
     tf.TensorSpec(model.inputs[1].shape, model.inputs[1].dtype)])

    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models",
                      name=pb_path,
                      as_text=False)


if __name__ == '__main__':
    # set_gpu(24)
    # left_rs_img_path = '/home/hesheng/Test/Tianjing/Dongli/GF7_DLC_E117.4_N39.2_20200319_L1A0000178402-BWDPAN.tiff'
    # right_rs_img_path = '/home/hesheng/Test/Tianjing/Dongli/GF7_DLC_E117.4_N39.2_20200319_L1A0000178402-FWDPAN.tiff'
    # height, width = 2048, 2048
    # pb_dir = '/home/hesheng/Networks/StereoNet/model/StereoNetMIX2048'
    # disp_path = '/home/hesheng/Test/Tianjing/Dongli/C1_GF7_DLC_E117.4_N39.2_20200319_L1A0000178402-BWDPAN_BF_LR_disp_1.tif'
    # predict_whole(left_rs_img_path, right_rs_img_path, height, width, pb_dir, disp_path)

    # model = tf.keras.models.load_model('../model/StereoNetMIX2048', compile=False)
    # convert_h5to_pb_two_input(model, 'StereoNet.pb')

    pass
