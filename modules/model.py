import tensorflow as tf
import numpy as np


def crop_2d(x, crop_size):
    start_idx = x.shape[1] // 2 - crop_size // 2
    stop_idx = start_idx + crop_size
    x = x[:, start_idx:stop_idx, start_idx:stop_idx, :]
    return x


def encoder_conv_crop_block(channels, name):
    def block(x):
        x = tf.keras.layers.Conv2D(channels, 3, use_bias=False, name=f'{name}_conv_1')(x)
        x = tf.keras.layers.LeakyReLU(0.01, name=f'{name}_relu_1')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bnorm_1')(x)
        x = tf.keras.layers.Conv2D(channels, 3, use_bias=False, name=f'{name}_conv_2')(x)
        x = tf.keras.layers.LeakyReLU(0.01, name=f'{name}_relu_2')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bnorm_2')(x)
        return x
    return block


def Encoder(levels, init_channels):
    def encoder(x_in):
        channels = init_channels
        outputs = list()
        x = inputs = tf.keras.layers.Input(x_in.shape[1:], name='encoder_input')
        for level in range(levels):
            x = x_skip = encoder_conv_crop_block(channels, f'encoder_conv_{level}')(x)
            outputs.append(x_skip)
            x = tf.keras.layers.MaxPool2D((2, 2), name=f'encoder_pool_{level}')(x)
            channels *= 2
        outputs.append(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name='Encoder')(x_in), channels
    return encoder


def Bridge(channels):
    def block(x_in):
        x = inputs = tf.keras.layers.Input(x_in.shape[1:], name='bridge_input')
        x = tf.keras.layers.Conv2D(channels, 3, use_bias=False, name='bridge_conv_1')(x)
        x = tf.keras.layers.LeakyReLU(0.01, name='bridge_relu_1')(x)
        x = tf.keras.layers.BatchNormalization(name='bridge_bnorm_1')(x)
        x = tf.keras.layers.Conv2D(channels, 3, use_bias=False, name='bridge_conv_2')(x)
        x = tf.keras.layers.LeakyReLU(0.01, name='bridge_relu_2')(x)
        x = tf.keras.layers.BatchNormalization(name='bridge_bnorm_2')(x)
        return tf.keras.Model(inputs=inputs, outputs=x, name='Bridge')(x_in)
    return block


def decoder_conv_block(channels, name):
    def block(x_in, x_skip):
        x_skip = tf.keras.layers.Lambda(lambda x: crop_2d(x, x_in.shape[1]), name=f'{name}_crop')(x_skip)
        x = tf.keras.layers.Concatenate(axis=-1, name=f'{name}_concat')([x_skip, x_in])
        x = tf.keras.layers.Conv2D(channels, 3, use_bias=False, name=f'{name}_conv_1')(x)
        x = tf.keras.layers.LeakyReLU(0.01, name=f'{name}_relu_1')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bnorm_1')(x)
        x = tf.keras.layers.Conv2D(channels, 3, use_bias=False, name=f'{name}_conv_2')(x)
        x = tf.keras.layers.LeakyReLU(0.01, name=f'{name}_relu_2')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bnorm_2')(x)
        return x
    return block


def Decoder(init_channels):
    def block(x_in, skip_connections):
        channels = init_channels
        inputs = [tf.keras.layers.Input(x_in.shape[1:], name='decoder_input_bridge')] + \
            [tf.keras.layers.Input(x_skip.shape[1:], name=f'decoder_input_skip_from_{level}')
             for level, x_skip in zip(range(len(skip_connections), -1, -1), reversed(skip_connections))]
        x = inputs[0]
        for level, x_skip in zip(range(len(skip_connections), -1, -1), inputs[1:]):
            x = tf.keras.layers.Conv2DTranspose(channels, 2, strides=2, padding='same', use_bias=False, name=f'decoder_deconv_{level}')(x)
            x = tf.keras.layers.LeakyReLU(0.01, name=f'decoder_relu_{level}')(x)
            x = tf.keras.layers.BatchNormalization(name=f'decoder_bnorm_{level}')(x)
            x = decoder_conv_block(channels, f'decoder_conv_{level}')(x, x_skip)
            channels = channels // 2
        model = tf.keras.Model(inputs=inputs, outputs=x, name='Decoder')([x_in, *list(reversed(skip_connections))])
        return model
    return block


def UNet(input_shape, levels, init_channels):
    x = inputs = tf.keras.layers.Input(input_shape, name='input')
    x, channels = Encoder(levels, init_channels)(x)
    skip_connections, x = x[:-1], x[-1]
    x = Bridge(channels)(x)
    x = Decoder(channels // 2)(x, skip_connections)
    x = outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='head_conv')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='UNet')


class ExportedModel:
    def __init__(self, model):
        self.model = model
    
    @tf.function
    def f(self, x):
        return self.model(x)


class InferenceModel:
    def __init__(self, model, weights_path, input_shape, output_shape, frame_shape, batch_size):
        model.load_weights(weights_path).expect_partial()
        self.model = ExportedModel(model)
        self.tile_dims = InferenceModel.reconstruction_dimensions(input_shape[1], output_shape[1], frame_shape)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.frame_shape = frame_shape
        self.batch_size = batch_size
        self.model.f(tf.random.normal((batch_size, *self.input_shape)))
        self.model.f(tf.random.normal((np.prod(self.tile_dims), *self.input_shape)))
    
    def preprocess(self, frame):
        frame = frame.astype(np.float32) / 255.
        input_size, output_size = self.input_shape[1], self.output_shape[1]
        input_padding = (input_size - output_size) // 2
        bot_output_padding = output_size - (frame.shape[0] % output_size) + input_padding
        right_output_padding = output_size - (frame.shape[1] % output_size) + input_padding

        paddings = tf.constant([[input_padding, bot_output_padding],
                                [input_padding, right_output_padding]])
        padded = tf.pad(frame, paddings, 'SYMMETRIC')[..., tf.newaxis]

        patches = list()
        row = input_padding
        for _ in range((padded.shape[0] - 2 * input_padding) // output_size):
            col = input_padding
            for _ in range((padded.shape[1] - 2 * input_padding) // output_size):
                patch_x = padded[row-input_padding:row+output_size+input_padding, col-input_padding:col+output_size+input_padding, ...]
                patches.append(patch_x)
                col += output_size
            row += output_size
        patches = tf.convert_to_tensor(patches, dtype=tf.float32)
        return patches

    def postprocess(self, predictions, norm_to_uint=False):
        class_map = np.zeros((self.tile_dims[0] * self.output_shape[0], self.tile_dims[1] * self.output_shape[1]), dtype=np.float32)
        row = 0
        for r in range(self.tile_dims[0]):
            col = 0
            for c in range(self.tile_dims[1]):
                predicted_map = predictions[r * self.tile_dims[1] + c, :, :, 0]
                class_map[row:row+self.output_shape[1], col:col+self.output_shape[1]] = predicted_map
                col += self.output_shape[1]
            row += self.output_shape[1]
        class_map = class_map[:self.frame_shape[0], :self.frame_shape[1]]
        if norm_to_uint:
            class_map = (class_map * 255).astype(np.uint8)
        return class_map

    def __call__(self, frame, norm_to_uint=False):
        preprocessed = self.preprocess(frame)
        predictions = list()
        for i in range(int(np.ceil(preprocessed.shape[0] / self.batch_size))):
            try:
                preprocessed_batch = preprocessed[i*self.batch_size:i*self.batch_size + self.batch_size]
            except IndexError:
                preprocessed_batch = preprocessed[i*self.batch_size:]
            predictions.append(self.model.f(preprocessed_batch))
        predictions = np.concatenate(predictions, axis=0)
        postprocessed = self.postprocess(predictions, norm_to_uint=norm_to_uint)
        return postprocessed

    @staticmethod
    def reconstruction_dimensions(input_size, output_size, frame_shape):
        input_padding = (input_size - output_size) // 2
        bot_output_padding = output_size - (frame_shape[0] % output_size) + input_padding
        right_output_padding = output_size - (frame_shape[1] % output_size) + input_padding

        new_shape = (frame_shape[0] + input_padding + bot_output_padding, frame_shape[1] + input_padding + right_output_padding)
        dims = ((new_shape[0] // output_size), (new_shape[1] // output_size))
        return dims


if __name__ == '__main__':
    shape = (256, 256, 1)
    model = UNet(shape, 3, 64)
    print(model(tf.random.normal((1, *shape))).shape)
