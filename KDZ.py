import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Запрос количества критериев у пользователя
num_criteria = int(input("Введите количество критериев: "))

# Запрос центров кластеров у пользователя
cluster_centers = []
for i in range(num_criteria):
    center = float(input(f"Введите центр для кластера {i + 1}: "))
    cluster_centers.append(center)
cluster_centers = np.array(cluster_centers)

# Чтение данных из Excel
df = pd.read_excel('Критерии.xlsx')

# Проверка, что в Excel-файле достаточно столбцов
if len(df.columns) < num_criteria + 1:  # +1 для учета столбца с номерами проектов
    raise ValueError(f"В Excel-файле недостаточно столбцов для {num_criteria} критериев.")

# Преобразование данных в массив numpy
criteria_columns = [f'f{i+1}' for i in range(num_criteria)]
projects = df[criteria_columns].to_numpy()

# Функция для определения парето-оптимальных точек
def find_pareto_optimal(criteria_values):
    num_projects = criteria_values.shape[0]
    is_pareto_optimal = np.ones(num_projects, dtype=bool)
    
    for i in range(num_projects):
        for j in range(num_projects):
            if i != j and np.all(criteria_values[j] >= criteria_values[i]) and np.any(criteria_values[j] > criteria_values[i]):
                is_pareto_optimal[i] = False
                break
    
    return is_pareto_optimal

# Определение парето-оптимальных точек
pareto_optimal = find_pareto_optimal(projects)
 
# Функция для вычисления индекса эффективности
def calculate_efficiency_index(criteria_values, pareto_optimal):
    num_projects = criteria_values.shape[0]
    efficiency_indices = np.zeros(num_projects)
    
    for i in range(num_projects):
            dominated_count = 0
            for j in range(num_projects):
                if i != j and np.all(criteria_values[j] >= criteria_values[i]):
                    dominated_count += 1
            efficiency_indices[i] = 1 / (1 + (dominated_count / (num_projects - 1)))
    return efficiency_indices

# Вычисление индексов эффективности
efficiency_indices = calculate_efficiency_index(projects, pareto_optimal)

# Функция для определения ближайшего кластера
def find_nearest_cluster(efficiency_index, cluster_centers):
    distances = np.abs(cluster_centers - efficiency_index)
    return np.argmin(distances) + 1

# Определение кластеров для каждого проекта
def assign_clusters(efficiency_indices, cluster_centers):
    clusters = []
    for efficiency_index in efficiency_indices:
        cluster = find_nearest_cluster(efficiency_index, cluster_centers)
        clusters.append(cluster)
    return clusters

# Определение кластеров для каждого проекта
clusters = assign_clusters(efficiency_indices, cluster_centers)

# Добавление индексов эффективности и кластеров в DataFrame
df['Индекс эффективности'] = efficiency_indices
df['Кластер'] = clusters

# Создание таблицы для вывода
output_table = df[['Индекс эффективности', 'Кластер']].copy()
for i in range(1, len(df) + 1):
    output_table[str(i)] = ''

for i, row in df.iterrows():
    if row['Индекс эффективности'] == 1:  # Оптимальные
        output_table.at[i, str(i + 1)] = '*'
    else:  # Заведомо неэффективные
        output_table.at[i, str(i + 1)] = 'x'

# Вывод окончательной таблицы
print("Окончательная таблица:")
print(output_table.to_string(index=False))
