# Главная функция
from math import log
from random import randint
from numpy import random
import numpy as np
import array as arr

class QAP:
    def __init__(self, assignments, distances,
                 flows):
        self.A = np.matrix(assignments) # Матрица стоимости назначений
        self.D = np.matrix(distances) # Матрица стоимости перевозки
        self.F = np.matrix(flows) # Количество единиц ресурса
        # IWO параметры
        self.maxIt = 50  # Максимальное количество итераций
        self.population_Size_Initial = 1  # Начальный размер популяции
        self.maximum_Population_Size = 10  # Максимальная размер популяции
        self.min_Seed = 0  # Минимальное количество семян
        self.max_Seed = 5  # Максимальное количество семян

        self.m = 2  # Показатель уменьшения дисперсии(m)

        self.sigma_initial = 0.5  # Начальное значение стандартного отклонения
        self.sigma_final = 0.001  # Конечное значение стандартного отклонения

    def start(self):

        # Генерируем исходную популяцию
        initial_population = np.array([random.uniform(-1,1) for i in range(len(self.A))])

        print(initial_population)
        indices = sorted(range(len(self.A)), key=lambda i: initial_population[i]) # Исходное решение



    def target_function(self, p):
        cost = 0
        for i in range(len(p)):  # i - объект, site - расположение
            cost += self.A[p[i], i]  # Находим сумма стоимости расположений объекта

        # f(i,j)*d(pi,pj)
        for i in range(1, len(p)):
            for j in range(i):
                cost += self.F[i, j] * self.D[p[i], p[j]]

        return cost







def iwo():
    # Размер матрицы задачи



    # Инициализируем случайным образом минимальное количество сорняков для старта алгоритма
    # Заполняем 0 гиперпараллелепипед
    initial_Population1 = np.zeros((population_Size_Initial,
                                    size_matrix_task,
                                    size_matrix_task),
                                   dtype=int)

    # Случайно заполняем 1(В данном случае по диагонали)
    for i in range(size_matrix_task):
        initial_Population1[0][i][i] = 1

    for i in range(size_matrix_task):
        initial_Population1[1][i][size_matrix_task - i - 1] = 1


    initial1 = dict()


    initial_Fitness1 = list()
    for i in range(population_Size_Initial):
        initial_Fitness1.append(target_function(matrix_task, initial_Population1[i]))
        initial1[matrix_in_tuple(initial_Population1[i])] = target_function(matrix_task, initial_Population1[i])

    print(initial_Fitness1)

    # Основной цикл
    for t in range(0, maxIt):
        # print(initial)
        # обновить стандартное отклонение по формуле
        sigma = (pow(((maxIt - t) / maxIt), m) * (sigma_initial - sigma_final)) + sigma_final

        best_Solution = min(initial_Fitness1)
        worst_Solution = max(initial_Fitness1)

        new_Initial_Population1 = list()
        new_Fitness_Population1 = list()

        new_Initial1 = dict()

        # фаза воспроизводства
        for i in range(0, len(initial1)):
            # Вычисляем число семян, которые может произвести данный сорняк
            ratio = (initial_Fitness1[i] - worst_Solution) / (best_Solution - worst_Solution)
            s = (min_Seed + ((max_Seed - min_Seed) * ratio))
            if s == 0:
                s += 1
            print(s)
            for j in range(0, round(s)):

                # Выбираем номера строк(столбцов) которые собираемся менять
                t1 = randint(0, size_matrix_task - 1)
                t2 = randint(0, size_matrix_task - 1)

                # Чтобы значения не совпадали
                while (t1 == t2):
                    t2 = randint(0, size_matrix_task - 1)

                # Меняем строки или столбцы
                row_or_column = randint(0, 1)  # Строка - 0, столбец - 1

                # Генерация потомка
                if (isinstance(initial_Population1, tuple)):
                    initial_Population1[i] = tuple_in_matrix(initial_Population1[i])

                # (0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0)
                print(initial_Population1[i])
                print()
                print(row_or_column, t1, t2)
                new_Solution_Position1 = mixing_matrix(initial_Population1[i], row_or_column, t1, t2)
                print_matrix(new_Solution_Position1)
                print()

                new_Solution_Cost1 = target_function(matrix_task, new_Solution_Position1)  # Вычисление целевой функции

                new_Initial_Population1.append(new_Solution_Position1)
                new_Fitness_Population1.append(new_Solution_Cost1)

                new_Initial1[matrix_in_tuple(new_Solution_Position1)] = new_Solution_Cost1

        # Объядинение популяции
        ini = initial1 | new_Initial1

        # Сортировка
        res = list(ini.items())
        res.sort(key=lambda x: x[1], reverse=False)

        initial_Population1 = list()
        initial_Fitness1 = list()
        initial1 = dict()
        j = 1

        # Исключаем слабых
        for i in res:
            if (j <= maximum_Population_Size):
                initial_Population1.append(i[0])
                initial_Fitness1.append(i[1])
                initial1[i[0]] = i[1]
            else:
                break
            j += 1
        print(initial1)
        print("Итерация : " + str(t) + "    Лучшее " + str(initial_Fitness1[0]))


def main():
    qap = QAP([[9, 51, 3], [2, 4, 1], [6, 22, 7]],[[0, 70, 2], [7, 0, 43], [2, 41, 0]],[[0, 31, 6], [3, 0, 42], [6, 4, 0]])
    qap.start()



if __name__ == '__main__':
    main()
