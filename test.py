import tensorflow as tf
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks, regularizers


class DynamicCosineDecay:
    def __init__(self, initial_lr, total_steps, alpha=0.01):
        self.initial_lr = tf.Variable(initial_lr, trainable=False)
        self.total_steps = total_steps
        self.alpha = alpha
        self.schedule = None

    def build_schedule(self):
        self.schedule = optimizers.schedules.CosineDecay(
            initial_learning_rate=self.initial_lr,
            decay_steps=self.total_steps,
            alpha=self.alpha
        )
        return self.schedule


def residual_block(x, filters, stride=1, first_block=False):
    shortcut = x
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    if not first_block:
        x = layers.BatchNormalization(axis=bn_axis)(x)
        x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride,
                                 kernel_regularizer=regularizers.l2(1e-4))(shortcut)

    x = layers.Add()([x, shortcut])
    return x


class BatchSizeScheduler(callbacks.Callback):
    def __init__(self, base_batch_size, max_batch_size, X_train):
        super().__init__()
        self.base_batch_size = base_batch_size
        self.max_batch_size = max_batch_size
        self.x_train = X_train
        self.current_batch_size = base_batch_size

    def on_epoch_begin(self, epoch, logs=None):
        # Gradually increase batch size
        new_size = min(self.max_batch_size, self.base_batch_size * (2 ** (epoch // 10)))
        if new_size != self.current_batch_size:
            self.current_batch_size = new_size
            self.model.optimizer.learning_rate.initial_lr.assign(1e-3 * (new_size / 128))

            self.model._train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, y_train)).map(lambda x, y: (data_augmentation(x), y),num_parallel_calls=AUTOTUNE).batch(new_size, drop_remainder=False).prefetch(AUTOTUNE)


def create_resnet():
    inputs = keras.Input(shape=(32, 32, 3))

    # Initial stem
    x = layers.Conv2D(64, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Stage 1
    x = residual_block(x, 64, first_block=True)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # Stage 2 (downsample)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    # Stage 3 (downsample)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    x = residual_block(x, 256)

    # Final
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(10, activation='softmax')(x)

    ret = keras.Model(inputs=inputs, outputs=outputs)

    # Cosine learning rate with warmup
    total_steps = 200 * (45000 // 128)
    lr_wrapper = DynamicCosineDecay(1e-3, total_steps)

    ret.compile(
        optimizer=optimizers.AdamW(learning_rate=lr_wrapper.build_schedule(), weight_decay=1e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return ret


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

batch_size = 128
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
])

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
).batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)

# Callbacks
callback_list = [
    BatchSizeScheduler(base_batch_size=128, max_batch_size=512, X_train=x_train),
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=30,
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=10,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath='old-models/best_resnet_model.keras',
        monitor='val_accuracy',
        save_best_only=True
    )
]

# Training
model = create_resnet()
print(model.summary())
