import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. Load dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# 2. Normalize images (0 to 1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# 3. Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 4. Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 5. Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. Train model
history = model.fit(train_images, train_labels,
                    epochs=5,
                    validation_data=(test_images, test_labels))

# 7. Evaluate model
loss, acc = model.evaluate(test_images, test_labels)
print("Test Accuracy:", acc)

# 8. Plot accuracy graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['Train', 'Validation'])
plt.savefig("accuracy_graph.png")
plt.show()


# 9. Detect an object from the test set (airplane = 0, cat = 3)
def detect_object(class_index):
    for i in range(len(test_labels)):
        if test_labels[i] == class_index:
            img = test_images[i]
            label = test_labels[i]
            break

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f"Test Image ({class_names[class_index].capitalize()})")
    plt.axis('off')
    plt.show()

    prediction = model.predict(np.expand_dims(img, axis=0))

    print("Actual:", class_names[label[0]])
    print("Predicted:", class_names[np.argmax(prediction)])


detect_object(0)  # airplane
detect_object(3)  # cat