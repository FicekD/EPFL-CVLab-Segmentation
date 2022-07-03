import tensorflow as tf

from modules.data_loader import init_2d_dataset
from modules.model import UNet
from config import CFG


def main():
    train_dataset, val_dataset = init_2d_dataset(
        CFG.DATASET_PATH['train']['x'],
        CFG.DATASET_PATH['train']['y'],
        validation_split_ratio=CFG.VALIDATION_SPLIT,
        input_size=CFG.INPUT_SHAPE[0],
        output_size=CFG.OUTPUT_SHAPE[0],
        batch_size=CFG.BATCH_SIZE,
        validate_on_patches=False,
        shuffle=True)

    model = UNet(CFG.INPUT_SHAPE, CFG.LEVELS, CFG.INIT_CHANNELS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=CFG.INIT_LEARNING_RATE)
    loss = tf.keras.losses.BinaryCrossentropy()
    callbacks = []

    model.compile(loss=loss, optimizer=optimizer)
    model.fit(train_dataset, validation_data=val_dataset, epochs=CFG.EPOCHS, callbacks=callbacks)

    model.save_weights(CFG.UNET_WEIGHTS)


if __name__ == '__main__':
    main()
