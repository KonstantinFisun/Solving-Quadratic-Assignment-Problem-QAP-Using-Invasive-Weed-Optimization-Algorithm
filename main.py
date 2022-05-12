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
        self.size = len(self.A)
        # IWO параметры
        self.maxIt = 50  # Максимальное количество итераций
        self.population_Size_Initial = 5  # Начальный размер популяции
        self.maximum_Population_Size = 10  # Максимальная размер популяции
        self.min_Seed = 0  # Минимальное количество семян
        self.max_Seed = 5  # Максимальное количество семян

        self.m = 2  # Показатель уменьшения дисперсии(m)

        self.sigma_initial = 0.5  # Начальное значение стандартного отклонения
        self.sigma_final = 0.001  # Конечное значение стандартного отклонения

    # Целевая функция
    def target_function(self, p):
        cost = 0
        for i in range(len(p)):  # i - объект, site - расположение
            cost += self.A[p[i], i]  # Находим сумма стоимости расположений объекта

        # f(i,j)*d(pi,pj)
        for i in range(1, len(p)):
            for j in range(i):
                cost += self.F[i, j] * self.D[p[i], p[j]]

        return cost

    # Основной алгоритм
    def start(self):

        population = list() # Начальная популяция
        # Генерируем начальную популяцию
        for i in range(self.population_Size_Initial):

            # Генерируем семя
            seed = np.array([random.uniform(-1,1) for i in range(len(self.A))]) # Равномерное распределение
            seed = sorted(range(len(self.A)), key=lambda i: seed[i]) # Исходное семя

            # Добавляем в популяцию, вычисляется целевая функция
            population.append((self.target_function(seed),tuple(seed)))

        population.sort(reverse=True) # Сортируем


        # Вычисления ведутся пока не не достигнуто конечное число операций
        for t in range(0, 1):
            # Обновить стандартное отклонение по формуле
            sigma = (pow(((self.maxIt - t) / self.maxIt), self.m) * (self.sigma_initial - self.sigma_final)) + self.sigma_final

            best_Solution = min(population)[0] # Лучшее значение
            worst_Solution = max(population)[0] # Худшее значение

            # фаза воспроизводства
            for i in range(0, len(population)):

                # Вычисляем число семян, которые может произвести данный сорняк
                ratio = (int(population[i][0]) - worst_Solution) / (best_Solution - worst_Solution)
                s = (self.min_Seed + ((self.max_Seed - self.min_Seed) * ratio))

                # Каждое семя
                for j in range(0, round(s)):
                    # Распределяем в окрестности родительского растения
                    print(np.random.normal(0, sigma, self.size))


            # # Объядинение популяции
            # ini = population
            #
            # # Сортировка
            # res = list(ini.items())
            # res.sort(key=lambda x: x[1], reverse=False)
            #
            #
            # # Исключаем слабых
            # for i in res:
            #     if (j <= maximum_Population_Size):
            #         initial_Population1.append(i[0])
            #         initial_Fitness1.append(i[1])
            #         initial1[i[0]] = i[1]
            #     else:
            #         break
            #     j += 1
            # print(initial1)
            # print("Итерация : " + str(t) + "    Лучшее " + str(initial_Fitness1[0]))

def main():
    qap = QAP([[9, 51, 3], [2, 4, 1], [6, 22, 7]],[[0, 70, 2], [7, 0, 43], [2, 41, 0]],[[0, 31, 6], [3, 0, 42], [6, 4, 0]])
    qap.start()



if __name__ == '__main__':
    main()
