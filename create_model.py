# Импорт модулей
import tensorflow as tf  # pip install tensorflow
import numpy as np  # pip install numpy
from PIL import Image  # pip install pillow

# Загрузка набора данных MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Предобработка данных
img_height, img_width = 28, 28  # Размеры изображений
x_train = x_train.astype('float32') / 255.0  # Нормализация
x_test = x_test.astype('float32') / 255.0  # Нормализация

# Убедимся, что данные имеют правильные формы
print(f'Количество train-изобр.: {x_train.shape[0]}, высота/ширина: {x_train.shape[1]}x{x_train.shape[2]}px')
print(f'Количество test-изобр.: {x_test.shape[0]}, высота/ширина: {x_test.shape[1]}x{x_test.shape[2]}px')

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

# Компиляция и обучение модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

# Оценка модели
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Точность на тестовом наборе данных: {test_acc}')

# Сохранение модели в новом формате Keras
model.save('my_model.keras')
print('Модель создана!')
