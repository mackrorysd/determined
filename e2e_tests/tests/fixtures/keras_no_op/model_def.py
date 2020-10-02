import numpy as np
import random
import tensorflow as tf
from determined.keras import TFKerasTrial, TFKerasTrialContext


class RandomMetric(tf.keras.metrics.Metric):
    def update_state(*args, **kwargs):
        pass

    def result(self):
        return random.randint(0, 1000)

class NumPyRandomMetric(tf.keras.metrics.Metric):
    def update_state(*args, **kwards):
        pass

    def result(self):
        return np.random.randint(1, 1000)

class NoopKerasTrial(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext):
        # Store trial context for later use.
        self.context = context

    def build_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(8, input_shape=(8,8,))
            ]
        )
        model = self.context.wrap_model(model)
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[
                RandomMetric(name="rand_rand"),
                NumPyRandomMetric(name="np_rand")
            ],
        )
        return model

    def build_training_data_loader(self):
        x_train = np.ones((64,8,8))
        y_train = np.ones((64,8,8))
        return (x_train, y_train)

    def build_validation_data_loader(self):
        x_val = np.ones((64,8,8))
        y_val = np.ones((64,8,8))
        return (x_val, y_val)
