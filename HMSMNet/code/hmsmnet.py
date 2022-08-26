import os
import glob
import numpy as np
from PIL import Image
import tensorflow.keras as keras
from feature import FeatureExtraction
from cost import CostConcatenation
from aggregation import Hourglass, FeatureFusion
from computation import Estimation
from refinement import Refinement
from scheduler import schedule
from loss_function import SmoothL1Loss
from data_reader import load_batch, read_left, read_right


class HMSMNet:
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
        gx = keras.Input(shape=(self.height, self.width, self.channel))
        gy = keras.Input(shape=(self.height, self.width, self.channel))

        feature_extraction = FeatureExtraction(filters=16)
        [l0, l1, l2] = feature_extraction(left_image)
        [r0, r1, r2] = feature_extraction(right_image)

        cost0 = CostConcatenation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        cost1 = CostConcatenation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        cost2 = CostConcatenation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)
        cost_volume0 = cost0([l0, r0])
        cost_volume1 = cost1([l1, r1])
        cost_volume2 = cost2([l2, r2])

        hourglass0 = Hourglass(filters=16)
        hourglass1 = Hourglass(filters=16)
        hourglass2 = Hourglass(filters=16)
        agg_cost0 = hourglass0(cost_volume0)
        agg_cost1 = hourglass1(cost_volume1)
        agg_cost2 = hourglass2(cost_volume2)

        estimator2 = Estimation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)
        disparity2 = estimator2(agg_cost2)

        fusion1 = FeatureFusion(units=16)
        fusion_cost1 = fusion1([agg_cost2, agg_cost1])
        hourglass3 = Hourglass(filters=16)
        agg_fusion_cost1 = hourglass3(fusion_cost1)

        estimator1 = Estimation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        disparity1 = estimator1(agg_fusion_cost1)

        fusion2 = FeatureFusion(units=16)
        fusion_cost2 = fusion2([agg_fusion_cost1, agg_cost0])
        hourglass4 = Hourglass(filters=16)
        agg_fusion_cost2 = hourglass4(fusion_cost2)

        estimator0 = Estimation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        disparity0 = estimator0(agg_fusion_cost2)

        # refinement
        refiner = Refinement(filters=32)
        final_disp = refiner([disparity0, left_image, gx, gy])

        self.model = keras.Model(inputs=[left_image, right_image, gx, gy],
                                 outputs=[disparity2, disparity1, disparity0, final_disp])
        self.model.summary()

    def build_single_output_model(self):
        left_image = keras.Input(shape=(self.height, self.width, self.channel))
        right_image = keras.Input(shape=(self.height, self.width, self.channel))
        gx = keras.Input(shape=(self.height, self.width, self.channel))
        gy = keras.Input(shape=(self.height, self.width, self.channel))

        feature_extraction = FeatureExtraction(filters=16)
        [l0, l1, l2] = feature_extraction(left_image)
        [r0, r1, r2] = feature_extraction(right_image)

        cost0 = CostConcatenation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        cost1 = CostConcatenation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        cost2 = CostConcatenation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)
        cost_volume0 = cost0([l0, r0])
        cost_volume1 = cost1([l1, r1])
        cost_volume2 = cost2([l2, r2])

        hourglass0 = Hourglass(filters=16)
        hourglass1 = Hourglass(filters=16)
        hourglass2 = Hourglass(filters=16)
        agg_cost0 = hourglass0(cost_volume0)
        agg_cost1 = hourglass1(cost_volume1)
        agg_cost2 = hourglass2(cost_volume2)

        estimator2 = Estimation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)
        disparity2 = estimator2(agg_cost2)

        fusion1 = FeatureFusion(units=16)
        fusion_cost1 = fusion1([agg_cost2, agg_cost1])
        hourglass3 = Hourglass(filters=16)
        agg_fusion_cost1 = hourglass3(fusion_cost1)

        estimator1 = Estimation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        disparity1 = estimator1(agg_fusion_cost1)

        fusion2 = FeatureFusion(units=16)
        fusion_cost2 = fusion2([agg_fusion_cost1, agg_cost0])
        hourglass4 = Hourglass(filters=16)
        agg_fusion_cost2 = hourglass4(fusion_cost2)

        estimator0 = Estimation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        disparity0 = estimator0(agg_fusion_cost2)

        # refinement
        refiner = Refinement(filters=32)
        final_disp = refiner([disparity0, left_image, gx, gy])

        self.model = keras.Model(inputs=[left_image, right_image, gx, gy], outputs=final_disp)
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
        mc = keras.callbacks.ModelCheckpoint(filepath=weights_save_path, monitor='refinement_loss',
                                             verbose=1, save_best_only=True, save_weights_only=True,
                                             mode='min', save_freq='epoch')

        optimizer = keras.optimizers.Adam()
        loss = [SmoothL1Loss(self.min_disp / 16.0, self.max_disp / 16.0),
                SmoothL1Loss(self.min_disp / 8.0, self.max_disp / 8.0),
                SmoothL1Loss(self.min_disp / 4.0, self.max_disp / 4.0),
                SmoothL1Loss(self.min_disp / 1.0, self.max_disp / 1.0)]
        loss_weights = [0.5, 0.7, 1.0, 0.6]

        if pre_trained_weights is not None:
            self.model.load_weights(filepath=pre_trained_weights, by_name=True)
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)
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
        left_image, gx, gy = read_left(os.path.join(left_dir, left))
        right_image = read_right(os.path.join(right_dir, right))
        left_image = np.expand_dims(left_image, 0)
        gx = np.expand_dims(gx, 0)
        gy = np.expand_dims(gy, 0)
        right_image = np.expand_dims(right_image, 0)
        disparity = model.predict([left_image, right_image, gx, gy])
        disparity = disparity[0, :, :, 0]
        disparity = Image.fromarray(disparity)
        name = left.replace('left', 'disparity')
        disparity.save(os.path.join(output_dir, name))
