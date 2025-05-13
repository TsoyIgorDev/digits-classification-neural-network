import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt  # Для отображения изображения

# Параметры
img_height = 28
img_width = 28

# Загрузка модели
model = tf.keras.models.load_model('my_model.keras')

# Функция для предсказания на изображении
def predict_digit(test_img):
    img = test_img.reshape(-1, img_height, img_width, 1)  # Изменение формы
    prediction = model.predict(img)
    return np.argmax(prediction)

# Пример использования с случайным изображением из тестового набора
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
random_index = np.random.randint(0, x_test.shape[0])  # Случайный индекс

# Получаем случайное изображение и соответствующую метку
test_image = x_test[random_index]
true_label = y_test[random_index]

# Прогнозируем цифру
predicted_digit = predict_digit(test_image)

# Отображаем результаты
plt.imshow(test_image, cmap='gray')
plt.title(f'Истинная цифра: {true_label}, Предсказанная цифра: {predicted_digit}')
plt.axis('off')
plt.show()
