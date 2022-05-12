# Главная функция
from math import log
from random import randint
from numpy import random
import numpy as np
import array as arr

class QAP:
    def __init__(self, distances,
                 flows):
        # self.A = np.matrix(assignments) # Матрица стоимости назначений
        self.D = np.matrix(distances) # Матрица стоимости перевозки
        self.F = np.matrix(flows) # Количество единиц ресурса
        self.size = len(self.D)
        # IWO параметры
        self.maxIt = 20  # Максимальное количество итераций
        self.population_Size_Initial = 20  # Начальный размер популяции
        self.maximum_Population_Size = 15  # Максимальная размер популяции
        self.min_Seed = 4  # Минимальное количество семян
        self.max_Seed = 8  # Максимальное количество семян

        self.m = 3  # Показатель уменьшения дисперсии(m)

        self.sigma_initial = 20  # Начальное значение стандартного отклонения
        self.sigma_final = 0.5  # Конечное значение стандартного отклонения

    # Целевая функция
    def target_function(self, p):
        cost = 0
        # for i in range(len(p)):  # i - объект, site - расположение
        #     cost += self.A[p[i], i]  # Находим сумма стоимости расположений объекта

        # f(i,j)*d(pi,pj)
        for i in range(1, len(p)):
            for j in range(i):
                cost += self.F[i, j] * self.D[p[i], p[j]]

        return cost*2

    # Основной алгоритм
    def start(self):

        population = list() # Начальная популяция
        # Генерируем начальную популяцию
        for i in range(self.population_Size_Initial):

            # Генерируем семя
            seed = np.array([random.uniform(-10,10) for i in range(len(self.D))]) # Равномерное распределение
            rand = seed
            seed = sorted(range(len(self.D)), key=lambda i: seed[i]) # Исходное семя

            # Добавляем в популяцию, вычисляется целевая функция
            population.append((self.target_function(seed),tuple(seed),tuple(rand)))

        population.sort(reverse=True) # Сортируем

        # Вычисления ведутся пока не не достигнуто конечное число операций
        for t in range(0, self.maxIt):
            # Обновить стандартное отклонение по формуле
            sigma = (pow(((self.maxIt - t) / self.maxIt), self.m) * (self.sigma_initial - self.sigma_final)) + self.sigma_final
            best_Solution = min(population)[0] # Лучшее значение
            worst_Solution = max(population)[0] # Худшее значение
            # фаза воспроизводства
            for i in range(0, len(population)):

                # Вычисляем число семян, которые может произвести данный сорняк
                ratio = float((int(population[i][0]) - int(worst_Solution)) / (int(best_Solution) - int(worst_Solution) + 1))
                s = (self.min_Seed + ((self.max_Seed - self.min_Seed) * ratio))

                # Каждое семя
                for j in range(0, round(s)):
                    # Распределяем в окрестности родительского растения
                    l = np.array(population[i][2])
                    k = np.random.normal(0, sigma, self.size)
                    seed = l + k
                    rand = seed
                    seed = sorted(range(len(self.D)), key=lambda i: seed[i])  # Семя

                    # Добавляем в популяцию, вычисляется целевая функция
                    population.append((self.target_function(seed), tuple(seed), tuple(rand)))

            # Сортировка
            population.sort()

            # Исключаем слабых
            population = population[0:self.maximum_Population_Size]
            #print(population[0:5])
            print("Итерация : " + str(t) + "    Лучшее " + str(population[0][0]) + str(population[0][1]))

    def stop(self):
        print(self.target_function([6, 4, 11,  1,  0,  2,  8, 10, 9,  5,  7,  3]))

def main():
    # qap = QAP([[9, 51, 3], [2, 4, 1], [6, 22, 7]],[[0, 70, 2], [7, 0, 43], [2, 41, 0]],[[0, 31, 6], [3, 0, 42], [6, 4, 0]])
    # qap.start()
    # n = 20
    # matrix_A = [[random.randint(1, 100) for j in range(n)] for i in range(n)]
    # matrix_B = [[random.randint(1, 100) for j in range(n)] for i in range(n)]
    # matrix_C = [[random.randint(1, 100) for j in range(n)] for i in range(n)]
    #
    # for i in range(n):
    #     for j in range(n):
    #         if i == j:
    #             matrix_B[i][j] = 0
    #             matrix_C[i][j] = 0

    data_F = []
    with open("chr12a(F).txt") as f:
        for line in f:
            data_F.append([int(x) for x in line.split()])
    data_D = []
    with open("chr12a(D).txt") as f:
        for line in f:
            data_D.append([int(x) for x in line.split()])
    qap = QAP(data_D, data_F)
    qap.start()
    qap.stop()



if __name__ == '__main__':
    main()
