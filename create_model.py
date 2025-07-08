# create_model.py с визуализацией обучения и EarlyStopping

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.callbacks import EarlyStopping

# Загрузка набора данных MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Предобработка данных
img_height, img_width = 28, 28
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Изменение формы данных для CNN (добавляем размерность канала)
x_train = x_train.reshape(-1, img_height, img_width, 1)
x_test = x_test.reshape(-1, img_height, img_width, 1)

# Создание модели нейросети
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(img_height, img_width, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Колбэк для ранней остановки обучения, если модель перестала улучшаться
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Обучение модели с валидацией и ранней остановкой
history = model.fit(
    x_train, y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=2
)

# Построение графиков точности и потерь
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Обучающая точность')
plt.plot(history.history['val_accuracy'], label='Валидационная точность')
plt.title('Точность модели')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Обучающая ошибка')
plt.plot(history.history['val_loss'], label='Валидационная ошибка')
plt.title('Ошибка модели')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()

plt.tight_layout()
plt.show()

# Оценка модели на тестовой выборке
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Точность на тестовом наборе данных: {test_acc}')

# Сохранение модели
model.save('my_model.keras')
print('Модель сохранена!')