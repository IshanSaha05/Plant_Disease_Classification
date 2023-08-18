#Importing libraries.
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML

#Setting all constants.
BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS=3
EPOCHS=50

#Importing dataset.
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/cse9040/Ishan/Dataset1/Data1/train",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

#Printing class names.
class_names = dataset.class_names
print("\nClass Names: ", class_names)   #38 classes present

#Splitting dataset.
"""
print("\nLength of the whole dataset: ", len(dataset))   #7030

train_size = 0.8
print("Length of the training set: ", len(dataset)*train_size)  #5624

train_ds = dataset.take(5624)
print("Length of the training set taken: ", len(train_ds))  #5624

test_ds = dataset.skip(5624)
print("\nSize of the remaining data: ", len(test_ds))  #1406

val_size=0.1
print("Size of the validation set: ", len(dataset)*val_size)  #140

val_ds = test_ds.take(140)
print("Size of the validation set taken: ", len(val_ds))  #140

test_ds = test_ds.skip(140)
print("Size of the test set taken: ", len(test_ds))  #1266
"""

#Splitting function definition.
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

#Finally splitting the dataset.
train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

#Finally prining the length of each dataset after the split.
print("\nLength of the training set: ", len(train_ds), "\nLength of the validation set: ", len(val_ds), "\nLength of the test set: ", len(test_ds))

#Cache, shuffle, prefetch is executed for better utilisation of the hardware.
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


#Asking tensorflow to allocate GPU memory accordingly.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)

#Creating a resizing and normalistion layer.
resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255),
])

#Doing data-augmentation.
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])

#Applying data-augmentation to dataset.
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y)).prefetch(buffer_size=tf.data.AUTOTUNE)

#Defining model architecture.
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 38

model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.build(input_shape=input_shape)

#Prining the model architecture.
print("\nModel Architecture: \n-------------------")
print(model.summary())

#Compiling the model.
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

#Fitting the model.
history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=50,
)

#Calculating the score.
scores = model.evaluate(test_ds)
print("\nScores: ", scores)
