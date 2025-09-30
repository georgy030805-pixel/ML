import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



data = pd.read_csv('titanic_train.csv', index_col='PassengerId')


def age_category(age):
    if age < 30:
        return 1
    elif age < 55:
        return 2
    else:
        return 3


data['Age_category'] = data['Age'].apply(age_category)

gender_counts = data['Sex'].value_counts()
print("Задание 1:")
print(f"Мужчины: {gender_counts['male']}")
print(f"Женщины: {gender_counts['female']}")
print(f"Правильный ответ: {gender_counts['male']} мужчин и {gender_counts['female']} женщин")

men_count = data[data['Sex'] == 'male'].shape[0]
women_count = data[data['Sex'] == 'female'].shape[0]
print(f"Проверка: мужчины - {men_count}, женщины - {women_count}")

print("\nЗадание 2:")

cross_table = pd.crosstab(data['Pclass'], data['Sex'])
print("Распределение по классам и полу:")
print(cross_table)

second_class_total = cross_table.loc[2].sum()
print(f"Всего людей во втором классе: {second_class_total}")

second_class_count = data[data['Pclass'] == 2].shape[0]
print(f"Проверка: {second_class_count}")

print(f"Правильный ответ: {second_class_total}")

print("\nЗадание 3:")

fare_median = data['Fare'].median()
fare_std = data['Fare'].std()

print(f"Медиана Fare: {fare_median:.2f}")
print(f"Стандартное отклонение Fare: {fare_std:.2f}")

answers = [
    "медиана 14,45, стандартное отклонение 49,69",
    "медиана 15,1, стандартное отклонение 12,15",
    "медиана 13,15, стандартное отклонение 35,3",
    "Медиана 17,43, стандартное отклонение - 39,1"
]

print(f"Правильный ответ: {answers[0]}")

print("\nЗадание 4:")

age_by_survival = data.groupby('Survived')['Age'].mean()
print("Средний возраст по группам выживания:")
print(f"Выжившие (Survived=1): {age_by_survival[1]:.2f} лет")
print(f"Погибшие (Survived=0): {age_by_survival[0]:.2f} лет")

is_survivors_older = age_by_survival[1] > age_by_survival[0]
print(f"Выжившие старше погибших: {'Да' if is_survivors_older else 'Нет'}")

print(f"Правильный ответ: {'Да' if is_survivors_older else 'Нет'}")

print("\nЗадание 5:")

young = data[data['Age'] < 30]
elderly = data[data['Age'] > 60]

young_survival_rate = young['Survived'].mean() * 100
elderly_survival_rate = elderly['Survived'].mean() * 100

print(f"Доля выживших среди молодежи (<30 лет): {young_survival_rate:.1f}%")
print(f"Доля выживших среди пожилых (>60 лет): {elderly_survival_rate:.1f}%")

answer_options = [
    "22,7% среди молодежи и 40,6% среди пожилых",
    "40,6% среди молодежи и 22,7% среди пожилых",
    "35,3% среди молодежи и 27,4% среди пожилых",
    "27,4% среди молодежи и 35,3% среди пожилых"
]

print(f"Правильный ответ: {answer_options[1]}")

print("\nЗадание 6:")

survival_by_gender = data.groupby('Sex')['Survived'].mean() * 100
print("Доли выживших по полу:")
print(f"Мужчины: {survival_by_gender['male']:.1f}%")
print(f"Женщины: {survival_by_gender['female']:.1f}%")

answer_options = [
    "30,2% среди мужчин и 46,2% среди женщин",
    "35,7% среди мужчин и 74,2% среди женщин",
    "21,1% среди мужчин и 46,2% среди женщин",
    "18,9% среди мужчин и 74,2% среди женщин"
]

print(f"Правильный ответ: {answer_options[3]}")

# Задание 7
male_passengers = data[data['Sex'] == 'male']
male_names = male_passengers['Name'].str.split(',').str[1].str.split('.').str[1].str.strip()
popular_name = male_names.value_counts().index[0]
print("Задание 7:")
print(f"Самое популярное мужское имя: {popular_name}")
print()

print("\nЗадание 8:")

age_analysis = data.groupby(['Pclass', 'Sex'])['Age'].mean()
print("Средний возраст по классам и полу:")
print(age_analysis)

print("\nПроверка утверждений:")

men_first_class_age = age_analysis.loc[(1, 'male')]
stmt1 = men_first_class_age > 40
print(f"1. Мужчины 1 класса старше 40 лет: {stmt1} ({men_first_class_age:.1f} лет)")

women_first_class_age = age_analysis.loc[(1, 'female')]
stmt2 = women_first_class_age > 40
print(f"2. Женщины 1 класса старше 40 лет: {stmt2} ({women_first_class_age:.1f} лет)")

age_comparison = data.groupby(['Pclass', 'Sex'])['Age'].mean().unstack()
stmt3 = (age_comparison['male'] > age_comparison['female']).all()
print(f"3. Мужчины всех классов старше женщин: {stmt3}")

avg_age_by_class = data.groupby('Pclass')['Age'].mean()
stmt4 = (avg_age_by_class[1] > avg_age_by_class[2]) and (avg_age_by_class[2] > avg_age_by_class[3])
print(f"4. 1 класс > 2 класс > 3 класс по возрасту: {stmt4}")
print(f"   Средний возраст по классам: 1кл={avg_age_by_class[1]:.1f}, 2кл={avg_age_by_class[2]:.1f}, 3кл={avg_age_by_class[3]:.1f}")

print(f"\nПравильные утверждения: 1, 4")
