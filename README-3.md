# Neural Networks and Deep Learning (NNDL) Lab

This repository contains 7 practical programs implemented using Python,
TensorFlow, Keras, Scikit-learn, NumPy and Matplotlib.

## Programs List

  -----------------------------------------------------------------------
  No.                     Program Name            Description
  ----------------------- ----------------------- -----------------------
  1                       CNN on MNIST Dataset    Recognizes handwritten
                                                  digits using CNN.

  2                       Digit Prediction        Predicts a digit and
                                                  displays prediction
                                                  probabilities.

  3                       Autoencoder             Reconstructs
                                                  handwritten digit
                                                  images.

  4                       Flower Classification   Classifies flower
                                                  images using CNN.

  5                       PCA                     Performs dimensionality
                                                  reduction using PCA.

  6                       LSTM                    Performs time-series
                                                  forecasting using LSTM.

  7                       RNN vs CNN              Compares the accuracy
                                                  of RNN and CNN models.
  -----------------------------------------------------------------------

## Requirements

Install the required libraries:

``` bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Program 1: CNN on MNIST Dataset

``` python
import tensorflow as tf
import matplotlib.pyplot as plt

(x, y), _ = tf.keras.datasets.mnist.load_data()

x = x[:1000].reshape(-1, 28, 28, 1) / 255.0
y = y[:1000]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(8, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, epochs=2, verbose=0)

plt.imshow(x[0].reshape(28, 28), cmap='gray')
plt.title("Character: " + str(y[0]))
plt.axis('off')
plt.show()
```

## Program 2: Digit Prediction

``` python
import tensorflow as tf
import matplotlib.pyplot as plt

(x, y), _ = tf.keras.datasets.mnist.load_data()

x = x[:1000].reshape(-1, 28, 28, 1) / 255.0
y = y[:1000]

m = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(8, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

m.compile(optimizer='adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

m.fit(x, y, epochs=2, verbose=0)

probabilities = m.predict(x[:1], verbose=0)
p = probabilities.argmax()

plt.subplot(1, 2, 1)
plt.imshow(x[0].reshape(28, 28), cmap='gray')
plt.title("Actual: " + str(y[0]))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(range(10), probabilities[0])
plt.title("Predicted: " + str(p))
plt.show()
```

## Program 3: Autoencoder

``` python
import tensorflow as tf
import matplotlib.pyplot as plt

(x, _), _ = tf.keras.datasets.mnist.load_data()

x = x[:1000].astype('float32') / 255
x = x.reshape(-1, 784)

m = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(784, activation='sigmoid')
])

m.compile(optimizer='adam', loss='mse')
m.fit(x, x, epochs=3, verbose=0)

out = m.predict(x[:1], verbose=0)

plt.subplot(1, 2, 1)
plt.imshow(x[0].reshape(28, 28), cmap='gray')
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(out[0].reshape(28, 28), cmap='gray')
plt.title("Reconstructed")

plt.show()
```

## Program 4: Flower Classification Using CNN

``` python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

ds = tf.keras.utils.image_dataset_from_directory(
    tf.keras.utils.get_file(
        "flower_photos",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
        untar=True
    ),
    image_size=(64, 64),
    batch_size=32
)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(5, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(ds, epochs=3)

images, labels = next(iter(ds))
img = images[0]
pred = model.predict(img[None, ...], verbose=0)

names = ds.class_names

plt.imshow(img.numpy().astype("uint8"))
plt.title("Predicted: " + names[np.argmax(pred)])
plt.axis("off")
plt.show()
```

## Program 5: PCA Dimensionality Reduction

``` python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target

p = PCA(n_components=2)
z = p.fit_transform(x)

plt.scatter(z[:, 0], z[:, 1], c=y)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Dimensionality Reduction")
plt.show()

print("Original shape:", x.shape)
print("Reduced shape:", z.shape)
```

## Program 6: LSTM Time Series Forecasting

``` python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x = np.sin(np.arange(100) * 0.1)

X = np.array([x[i:i + 5] for i in range(90)])
Y = x[5:95]

X = X.reshape(90, 5, 1)

m = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5, 1)),
    tf.keras.layers.LSTM(10),
    tf.keras.layers.Dense(1)
])

m.compile(optimizer='adam', loss='mse')
m.fit(X, Y, epochs=10, verbose=0)

pred = m.predict(X, verbose=0).flatten()

plt.plot(Y, label="Actual")
plt.plot(pred, label="Predicted")
plt.title("LSTM Time Series Forecasting")
plt.legend()
plt.show()
```

## Program 7: RNN vs CNN Comparison

``` python
import tensorflow as tf
import matplotlib.pyplot as plt

(x, y), _ = tf.keras.datasets.mnist.load_data()

x = x[:1000].astype('float32') / 255
y = y[:1000]

rnn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(10, activation='softmax')
])

cnn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(8, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

rnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

rnn.fit(x, y, epochs=2, verbose=0)
cnn.fit(x[..., None], y, epochs=2, verbose=0)

a = rnn.evaluate(x, y, verbose=0)[1]
b = cnn.evaluate(x[..., None], y, verbose=0)[1]

plt.bar(["RNN", "CNN"], [a, b])
plt.ylabel("Accuracy")
plt.title("RNN vs CNN")
plt.show()

print("RNN accuracy:", round(a * 100, 2), "%")
print("CNN accuracy:", round(b * 100, 2), "%")
```

## Technologies Used

-   Python
-   TensorFlow and Keras
-   NumPy
-   Matplotlib
-   Scikit-learn
-   Convolutional Neural Networks (CNN)
-   Recurrent Neural Networks (RNN)
-   Long Short-Term Memory (LSTM)
-   Principal Component Analysis (PCA)

## How to Run

1.  Clone this repository.
2.  Install the required libraries.
3.  Open the programs using Jupyter Notebook, Google Colab or VS Code.
4.  Execute each program separately.

``` bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
pip install -r requirements.txt
```

## Suggested Repository Structure

``` text
NNDL-LAB/
├── README.md
├── requirements.txt
├── Program-1-CNN-MNIST.py
├── Program-2-Digit-Prediction.py
├── Program-3-Autoencoder.py
├── Program-4-Flower-Classification.py
├── Program-5-PCA.py
├── Program-6-LSTM.py
└── Program-7-RNN-vs-CNN.py
```

## Author

**Karthik Reddy**\
B.Tech -- Computer Science and Engineering\
The Apollo University

------------------------------------------------------------------------

*This repository is created for academic learning and Neural Networks
and Deep Learning laboratory practice.*
