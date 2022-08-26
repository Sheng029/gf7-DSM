import os
import glob
import numpy as np
from PIL import Image
from component import *
from data_reader import read_image, load_batch
from schedule import schedule
from loss_function import SmoothL1Loss


class PSMNet:
    def __init__(self, height, width, channel, min_disp, max_disp):
        self.height = height
        self.width = width
        self.channel = channel
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.model = None

    def build_model(self):
        left_image = keras.Input(shape=(self.height, self.width, self.channel))
        right_image = keras.Input(shape=(self.height, self.width, self.channel))

        extractor = FeatureExtractor(filters=32)
        left_feature = extractor(left_image)
        right_feature = extractor(right_image)

        constructor = CostVolume(self.min_disp // 4, self.max_disp // 4)
        cost_volume = constructor([left_feature, right_feature])

        hourglass = StackedHourglass(filters=32)
        [out1, out2, out3] = hourglass(cost_volume)

        estimation = Estimation(self.min_disp, self.max_disp)
        d1 = estimation(out1)
        d2 = estimation(out2)
        d3 = estimation(out3)

        self.model = keras.Model(inputs=[left_image, right_image], outputs=[d1, d2, d3])
        self.model.summary()

    def build_single_output_model(self):
        left_image = keras.Input(shape=(self.height, self.width, self.channel))
        right_image = keras.Input(shape=(self.height, self.width, self.channel))

        extractor = FeatureExtractor(filters=32)
        left_feature = extractor(left_image)
        right_feature = extractor(right_image)

        constructor = CostVolume(self.min_disp // 4, self.max_disp // 4)
        cost_volume = constructor([left_feature, right_feature])

        hourglass = StackedHourglass(filters=32)
        [out1, out2, out3] = hourglass(cost_volume)

        estimation = Estimation(self.min_disp, self.max_disp)
        d1 = estimation(out1)
        d2 = estimation(out2)
        d3 = estimation(out3)

        self.model = keras.Model(inputs=[left_image, right_image], outputs=d3)
        self.model.summary()

    def train_only(self, data_dir, weights_save_path, epochs, batch_size, pre_trained_weights):
        # all paths
        all_left_paths = glob.glob(data_dir + '/left/*')
        all_right_paths = glob.glob(data_dir + '/right/*')
        all_disp_paths = glob.glob(data_dir + '/disp/*')

        # sort, necessary in Linux
        all_left_paths.sort()
        all_right_paths.sort()
        all_disp_paths.sort()

        # callbacks
        lr = keras.callbacks.LearningRateScheduler(schedule=schedule, verbose=1)
        mc = keras.callbacks.ModelCheckpoint(filepath=weights_save_path, monitor='estimation_2_loss',
                                             verbose=1, save_best_only=True, save_weights_only=True,
                                             mode='min', save_freq='epoch')

        optimizer = keras.optimizers.Adam()
        loss = [SmoothL1Loss(1.0 * self.min_disp, 1.0 * self.max_disp),
                SmoothL1Loss(1.0 * self.min_disp, 1.0 * self.max_disp),
                SmoothL1Loss(1.0 * self.min_disp, 1.0 * self.max_disp)]
        loss_weights = [0.5, 0.7, 1.0]

        if pre_trained_weights is not None:
            self.model.load_weights(filepath=pre_trained_weights, by_name=True)
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.fit_generator(
            generator=load_batch(all_left_paths, all_right_paths, all_disp_paths, batch_size, True),
            steps_per_epoch=len(all_disp_paths) // batch_size, epochs=epochs, callbacks=[lr, mc],
            shuffle=False)


def predict(left_dir, right_dir, output_dir, model_dir):
    model = keras.models.load_model(model_dir, compile=False)
    lefts = os.listdir(left_dir)
    rights = os.listdir(right_dir)
    lefts.sort()
    rights.sort()
    assert len(lefts) == len(rights)
    for left, right in zip(lefts, rights):
        left_image = read_image(os.path.join(left_dir, left))
        right_image = read_image(os.path.join(right_dir, right))
        left_image = np.expand_dims(left_image, 0)
        right_image = np.expand_dims(right_image, 0)
        disparity = model.predict([left_image, right_image])
        disparity = disparity[0, :, :, 0]
        disparity = Image.fromarray(disparity)
        name = left.replace('left', 'disparity')
        disparity.save(os.path.join(output_dir, name))
