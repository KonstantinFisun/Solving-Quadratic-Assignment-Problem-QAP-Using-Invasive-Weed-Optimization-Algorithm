# Главная функция
from math import log
from random import randint
from numpy import random
import numpy as np
import array as arr


class QAP:
    def __init__(self, distances,
                 flows, maxIt = 20, population_Size_Initial = 20,
                 maximum_Population_Size = 20,min_Seed = 1, max_Seed = 3, m = 3, sigma_initial = 20, sigma_final = 0.5):
        # self.A = np.matrix(assignments) # Матрица стоимости назначений
        self.D = np.matrix(distances)  # Матрица стоимости перевозки
        self.F = np.matrix(flows)  # Количество единиц ресурса
        self.size = len(self.D)
        # IWO параметры
        self.maxIt = maxIt  # Максимальное количество итераций
        self.population_Size_Initial = population_Size_Initial  # Начальный размер популяции
        self.maximum_Population_Size = maximum_Population_Size  # Максимальная размер популяции
        self.min_Seed = min_Seed  # Минимальное количество семян
        self.max_Seed = max_Seed  # Максимальное количество семян

        self.m = m  # Показатель уменьшения дисперсии(m)

        self.sigma_initial = sigma_initial  # Начальное значение стандартного отклонения
        self.sigma_final = sigma_final  # Конечное значение стандартного отклонения

    # Целевая функция
    def target_function(self, p):
        cost = 0
        # for i in range(len(p)):  # i - объект, site - расположение
        #     cost += self.A[p[i], i]  # Находим сумма стоимости расположений объекта
        # f(i,j)*d(pi,pj)
        for i in range(1, len(p)):
            for j in range(i):
                cost += self.F[i, j] * self.D[p[i], p[j]]

        return cost * 2

    # Основной алгоритм
    def start(self):

        population = list()  # Начальная популяция
        # Генерируем начальную популяцию
        for i in range(self.population_Size_Initial):
            # Генерируем семя
            seed = np.array([random.uniform(-5, 5) for i in range(len(self.D))])  # Равномерное распределение
            rand = seed
            seed = sorted(range(len(self.D)), key=lambda i: seed[i])  # Исходное семя

            # Добавляем в популяцию, вычисляется целевая функция
            population.append((self.target_function(seed), tuple(seed), tuple(rand)))

        population.sort(reverse=True)  # Сортируем

        # Вычисления ведутся пока не не достигнуто конечное число операций
        for t in range(0, self.maxIt):
            # Обновить стандартное отклонение по формуле
            sigma = (pow(((self.maxIt - t) / self.maxIt), self.m) * (
                    self.sigma_initial - self.sigma_final)) + self.sigma_final
            best_Solution = min(population)[0]  # Лучшее значение
            worst_Solution = max(population)[0]  # Худшее значение
            # фаза воспроизводства
            for i in range(0, len(population)):

                # Вычисляем число семян, которые может произвести данный сорняк
                ratio = float(
                    (int(population[i][0]) - int(worst_Solution)) / (int(best_Solution) - int(worst_Solution) + 1))
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
            # print(population[0:5])
            print("Итерация : " + str(t) + "    Лучшее " + str(population[0][0]) + str(population[0][1]))

    def stop(self):
        print(self.target_function([7, 15, 13, 16, 3, 10, 2, 18, 6, 8, 0, 14, 5, 12, 9, 1, 4, 19, 17, 11]))


def main():
    # получим объект файла
    print()
    fil = input("Введите название входного файла с матрицами: ")
    file1 = open(f"{fil}.txt", "r")
    data_F = []
    data_D = []

    # считываем строку
    size = file1.readline()
    file1.readline()
    for i in range(int(size)):
        data_F.append([int(x) for x in file1.readline().split()])
    for i in range(int(size)):
        data_D.append([int(x) for x in file1.readline().split()])
    # print(data_D)
    # print(data_F)
    # закрываем файл
    file1.close()

    # начальный и максимальный размер популяции,
    # минимальное и максимальное число семян,
    # показатель уменьшения дисперсии,
    # начальное и конечное значение стандартного отклонения
    # количество итераций
    maxIt = int(input("Количество итераций: "))
    population_Size_Initial = int(input("Начальный размер популяции: "))
    maximum_Population_Size = int(input("Максимальный размер популяции: "))
    min_Seed = int(input("Минимальное число семян: "))
    max_Seed = int(input("Максимальное число семян: "))
    m = int(input("Показатель уменьшения дисперсии: "))
    sigma_initial = int(input("Начальное значение стандартного отклонения:"))
    sigma_final = int(input("Конечное значение стандартного отклонения:"))
    qap = QAP(data_D, data_F, maxIt, population_Size_Initial,
              maximum_Population_Size,min_Seed, max_Seed,
              m, sigma_initial, sigma_final)
    qap.start()



if __name__ == '__main__':
    main()
