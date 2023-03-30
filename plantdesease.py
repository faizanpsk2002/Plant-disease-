import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16


BATCH_SIZE = 32
IMAGE_SIZE = 220
CHANNELS=3
EPOCHS=10


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)



class_names = dataset.class_names
class_names


for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())



plt.figure(figsize=(14, 10))
for image_batch, labels_batch in dataset.take(1):
    for i in range(15):
        plt.subplot(5, 3, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.xticks()
        plt.show()


