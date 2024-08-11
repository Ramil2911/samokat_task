"""
Этот файл содержит иерархический классификатор, написанный для тестового залания в ecom.tech

Базовый классификатор обучается для каждого отдельного родительского класса (то есть имеющего дочерние). Один родительский класс - один классификатор.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

class TopDownClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier, hierarchy, config=None):
        """
        Инициализация TopDownClassifier.

        :param base_classifier: Базовый классификатор, который будет использоваться на каждом уровне иерархии.
        :param hierarchy: Иерархия классов в виде вложенного словаря.
        :param config: Дополнительные параметры, которые будут переданы базовому классификатору.
        """
        self.base_classifier = base_classifier
        self.hierarchy = hierarchy
        self.config = config if config is not None else {}
        self.models = {}  # Словарь для хранения обученных моделей на каждом уровне
        self.encoders = {}  # Словарь для хранения LabelEncoder для каждого уровня

    def fit(self, X, y):
        """
        Обучение модели на данных X и многомерных метках y.

        :param X: Входные данные, матрица признаков.
        :param y: Двумерный массив меток классов, где каждый столбец соответствует уровню иерархии.
        :return: self
        """
        self._fit_recursive(X, y, current_path=('root',), current_branch=self.hierarchy, level=0)
        return self

    def _fit_recursive(self, X, y, current_path, current_branch, level):
        """
        Рекурсивная функция для обучения моделей на каждом уровне иерархии.

        :param X: Входные данные, матрица признаков.
        :param y: Двумерный массив меток классов, где каждый столбец соответствует уровню иерархии.
        :param current_path: Текущий путь в иерархии (кортеж).
        :param current_branch: Текущая ветвь иерархии (словарь).
        :param level: Текущий уровень иерархии.
        """

        # Кодируем метки на текущем уровне с помощью LabelEncoder
        encoder_key = current_path[-1]
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y[:, level])
        self.encoders[current_path] = encoder

        # Если на текущем уровне нет подкатегорий, завершить рекурсию
        if len(current_branch[encoder_key]) == 1:
            return

        # Обучаем модель на текущем уровне
        model = self.base_classifier(**self.config)
        model.fit(X, y_encoded)
        self.models[current_path] = model

        # Для каждого подкласса запускаем рекурсивное обучение
        unique_classes = encoder.classes_
        for class_idx, sub_class in enumerate(unique_classes):
            sub_class_indices = y_encoded == class_idx
            if np.any(sub_class_indices):
                new_path = current_path + (sub_class,)
                if sub_class in current_branch[encoder_key] and len(current_branch[encoder_key][sub_class]) > 0:
                    self._fit_recursive(
                        X[sub_class_indices],
                        y[sub_class_indices],
                        new_path,
                        current_branch[encoder_key],
                        level + 1
                    )

    def predict(self, X):
        """
        Предсказание классов для входных данных X.

        :param X: Входные данные, матрица признаков.
        :return: Двумерный массив предсказанных меток классов.
        """
        predictions = [self._predict_recursive(x, ('root',), self.hierarchy) for x in X]
        return np.array(predictions)

    def _predict_recursive(self, x, current_path, current_branch, level=0):
        """
        Рекурсивная функция для предсказания класса на каждом уровне иерархии.

        :param x: Один образец данных (одномерный массив признаков).
        :param current_path: Текущий путь в иерархии (кортеж).
        :param current_branch: Текущая ветвь иерархии (словарь).
        :param level: Текущий уровень в иерархии.
        :return: Список предсказанных меток на всех уровнях иерархии.
        """

        encoder_key = current_path[-1]

        # Проверка, существует ли кодировщик для текущего уровня
        if current_path not in self.encoders:
            return []

        # Если на текущем уровне нет подкатегорий, возвращаем единственный класс
        if len(current_branch[encoder_key]) == 1:
            return [self.encoders[current_path].inverse_transform([0])[0]]

        # Получаем предсказание модели на текущем уровне
        model = self.models[current_path]
        pred_encoded = model.predict(x.reshape(1, -1))[0]
        pred_class = self.encoders[current_path].inverse_transform([pred_encoded])[0]
        predictions = [pred_class]

        # Если предсказанный класс имеет подкатегории, продолжаем предсказание рекурсивно
        new_path = current_path + (pred_class,)
        if pred_class in current_branch[encoder_key] and len(current_branch[encoder_key][pred_class]) > 0:
            predictions.extend(self._predict_recursive(
                x, new_path, current_branch[encoder_key], level + 1
            ))

        return predictions