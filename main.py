# Главная функция
from math import log
from random import randint
from numpy import random
import numpy as np
import array as arr

size_matrix_task = 5  # Размер исходной матрицы

# Матрица затрат С
matrix_task = [[49, 74, 62, 80, 58],
               [91, 73, 67, 32, 31],
               [11, 85, 15, 8, 64],
               [55, 41, 47, 15, 74],
               [83, 87, 30, 13, 78]]

# Оптимальное решение X
x = [
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
]


# Целевая функция
def target_function(c, x):
    result = 0  # Результат

    for i in range(len(c)):
        for j in range(len(c)):
            result += c[i][j] * x[i][j]

    return result


# Вывод матриц
def print_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            print("{:4d}".format(matrix[i][j]), end="")
        print()

def mixing_matrix(matrix, row_or_column, pos1, pos2):
    # Меняем строки
    if (isinstance(matrix, tuple)):
        matrix = tuple_in_matrix(matrix)

    size_matrix = size_matrix_task

    if row_or_column == 0:
        for i in range(size_matrix):
            t = matrix[pos1][i]
            matrix[pos1][i] = matrix[pos2][i]
            matrix[pos2][i] = t


    # Меняем столбцы
    else:
        for i in range(size_matrix):
            t = matrix[i][pos1]
            matrix[i][pos1] = matrix[i][pos2]
            matrix[i][pos2] = t

    return matrix

# Из матрицы делаем кортеж
def matrix_in_tuple(matrix):
    a = []
    for i in range(len(matrix)):
        a.append([])
        for j in range(len(matrix)):
            a[i].append(matrix[i][j])

    lis = list()
    for i in range(0, len(a)):
        for j in range(0, len(a)):
            lis.append(a[i][j])

    return tuple(lis)


# Из картежа делаем матрицу
def tuple_in_matrix(tuple):
    matrix_list = list(tuple)  # Получаем список
    k = 0
    matrix = []
    for i in range(size_matrix_task):
        a = []
        for j in range(size_matrix_task):
            a.append(matrix_list[k])
            k += 1
        matrix.append(a)

    return np.array(matrix)


def iwo():
    # Размер матрицы задачи

    # IWO параметры
    maxIt = 50  # Максимальное количество итераций
    population_Size_Initial = 2  # Начальный размер популяции
    maximum_Population_Size = 10  # Максимальная размер популяции
    min_Seed = 0  # Минимальное количество семян
    max_Seed = 5  # Максимальное количество семян

    m = 2  # Показатель уменьшения дисперсии(m)

    sigma_initial = 0.5  # Начальное значение стандартного отклонения
    sigma_final = 0.001  # Конечное значение стандартного отклонения

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
    iwo()
    return


if __name__ == '__main__':
    main()
