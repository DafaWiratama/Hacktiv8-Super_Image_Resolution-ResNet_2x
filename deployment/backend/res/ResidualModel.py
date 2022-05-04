import tensorflow as tf
import cv2


class ResidualModel:
    N_UNITS = 64
    KERNEL_SIZE = (3, 3)

    def __init__(self, path):
        self.weight_mse_path = f'{path}/generator_mse.h5'
        self.weight_gan_path = f'{path}/generator_gan.h5'
        self.weight_perceptual_path = f'{path}/generator_perceptual.h5'
        self.model = self.create_model()

    def create_model(self):
        _input = tf.keras.layers.Input(shape=(None, None, 3))
        x = tf.keras.layers.Rescaling(1 / 255.)(_input)
        _x = x = tf.keras.layers.Conv2D(self.N_UNITS, (3, 3), padding='same')(x)

        for i in range(16):
            _x = self.__residual_block(_x, self.N_UNITS, self.KERNEL_SIZE, padding='same')

        _x = tf.keras.layers.Conv2D(self.N_UNITS, self.KERNEL_SIZE, padding='same')(_x)
        _x = tf.keras.layers.BatchNormalization()(_x)
        x = tf.keras.layers.Add()([x, _x])

        x = tf.keras.layers.Conv2D(self.N_UNITS * 4, self.KERNEL_SIZE, padding='same')(x)
        x = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

        x = tf.keras.layers.Conv2D(3, (1, 1), padding='same', activation='tanh')(x)
        x = tf.keras.layers.Lambda(lambda x: (x + 1.) * 127.5)(x)
        return tf.keras.models.Model(_input, x, name=f"Generator")

    @staticmethod
    def __residual_block(x, units, filter, padding):
        _x = tf.keras.layers.Conv2D(units, filter, padding=padding)(x)
        _x = tf.keras.layers.BatchNormalization(momentum=.8)(_x)
        _x = tf.keras.layers.PReLU(shared_axes=[1, 2])(_x)

        _x = tf.keras.layers.Conv2D(units, filter, padding=padding)(_x)
        _x = tf.keras.layers.BatchNormalization(momentum=.8)(_x)

        x = tf.keras.layers.Add()([x, _x])
        return x

    def __call__(self, x, weights, denoiser=True):
        if weights == 'mse':
            self.model.load_weights(self.weight_mse_path)
        elif weights == 'perceptual':
            self.model.load_weights(self.weight_perceptual_path)
        else:
            self.model.load_weights(self.weight_gan_path)
        x = self.model(x)
        x = tf.cast(x, tf.uint8)[0]

        if denoiser:
            x = self.__denoiser(x)
        return x

    @staticmethod
    def __denoiser(image):
        print("Denoise")
        image = image.numpy()
        image = cv2.fastNlMeansDenoisingColored(image, None, h=5, hColor=3, templateWindowSize=21, searchWindowSize=7)
        return tf.cast(image, tf.uint8)
