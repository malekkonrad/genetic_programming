import os
import numpy as np
import matplotlib.pyplot as plt
from typing import *

def generate_data(draw_charts: Optional[bool] = False):
    
    PATH = './data'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    def generate_data_file(file_name: str, var_num: int, formula: callable, domain_start: float, domain_end: float, num_of_points: Optional[int] = 100):
        if var_num == 1:
            points = np.linspace(domain_start, domain_end, num=num_of_points)
            y_ = formula(points)
            
            # extra część rysująca wykresy
            if draw_charts:
                fig = plt.figure()
                plt.plot(points, y_)
                plt.title(f'File name: {file_name} in [{domain_start}, {domain_end}]')
                plt.show()

            with open(file_name, 'w') as file:
                file.write(f'{var_num} 100 -5 5 {num_of_points}\n')
                for x in points:
                    y = formula(x)
                    file.write(f'{x} {y}\n')

        if var_num == 2:

            x = np.linspace(domain_start, domain_end, num=num_of_points)
            y = np.linspace(domain_start+1, domain_end+1, num=num_of_points)
            xv, yv = np.meshgrid(x, y)
            xv, yv = xv.flatten(), yv.flatten()
            z = formula(xv, yv)


            # extra część - z robieniem wykresów
            if draw_charts:
                fig = plt.figure()
                ax = plt.axes(projection='3d')

                ax.plot3D(xv, yv, z, 'green')
                ax.set_title(f'File name: {file_name} in [{domain_start}, {domain_end}]')
                plt.show()

            with open(file_name, 'w') as file:
                file.write(f'{var_num} 100 5 -5 {num_of_points*num_of_points}\n')
                for x_, y_, z_ in zip(xv, yv, z):
                    file.write(f'{x_} {y_} {z_}\n')

    # 1
    generate_data_file('./data/problem1_a.dat', 1, lambda x: 5*x**3 - 2*x**2 + 3*x - 17, -10, 10, num_of_points=100)
    generate_data_file('./data/problem1_b.dat', 1, lambda x: 5*x**3 - 2*x**2 + 3*x - 17, 0, 100, num_of_points=100)
    generate_data_file('./data/problem1_c.dat', 1, lambda x: 5*x**3 - 2*x**2 + 3*x - 17, -1, 1, num_of_points=100)
    generate_data_file('./data/problem1_d.dat', 1, lambda x: 5*x**3 - 2*x**2 + 3*x - 17, -1000, 1000, num_of_points=100)

    # 2
    generate_data_file('./data/problem2_a.dat', 1, lambda x: np.sin(x) + np.cos(x), -3.14, 3.14, num_of_points=100)
    generate_data_file('./data/problem2_b.dat', 1, lambda x: np.sin(x) + np.cos(x), 0, 7, num_of_points=100)
    generate_data_file('./data/problem2_c.dat', 1, lambda x: np.sin(x) + np.cos(x), 0, 100, num_of_points=100)
    generate_data_file('./data/problem2_d.dat', 1, lambda x: np.sin(x) + np.cos(x), -100, 100, num_of_points=100)   # ciekawa!!!

    # # 3
    generate_data_file('./data/problem3_a.dat', 1, lambda x: 2* np.log(x+1), 0, 4, num_of_points=100)
    generate_data_file('./data/problem3_b.dat', 1, lambda x: 2* np.log(x+1), 0, 9, num_of_points=100)
    generate_data_file('./data/problem3_c.dat', 1, lambda x: 2* np.log(x+1), 0, 99, num_of_points=100)
    generate_data_file('./data/problem3_d.dat', 1, lambda x: 2* np.log(x+1), 0, 999, num_of_points=100)

    # 4
    generate_data_file('./data/problem4_a.dat', 2, lambda x, y: x + 2*y , 0, 1, num_of_points=100)
    generate_data_file('./data/problem4_b.dat', 2, lambda x, y: x + 2*y , -10, 10, num_of_points=100)
    generate_data_file('./data/problem4_c.dat', 2, lambda x, y: x + 2*y , 0, 100, num_of_points=100)
    generate_data_file('./data/problem4_d.dat', 2, lambda x, y: x + 2*y , -1000, 1000, num_of_points=100)

    # 5
    generate_data_file('./data/problem5_a.dat', 1, lambda x: np.sin(x/2) + 2*np.cos(x), -3.14, 3.14, num_of_points=100)
    generate_data_file('./data/problem5_b.dat', 1, lambda x: np.sin(x/2) + 2*np.cos(x), 0, 7, num_of_points=100)
    generate_data_file('./data/problem5_c.dat', 1, lambda x: np.sin(x/2) + 2*np.cos(x), 0, 100, num_of_points=100)
    generate_data_file('./data/problem5_d.dat', 1, lambda x: np.sin(x/2) + 2*np.cos(x), -100, 100, num_of_points=100)   

    # 6
    generate_data_file('./data/problem6_a.dat', 2, lambda x, y: x**2 + 3*x*y - 7*y + 1, -10, 10, num_of_points=100)
    generate_data_file('./data/problem6_b.dat', 2, lambda x, y: x**2 + 3*x*y - 7*y + 1, 0, 100, num_of_points=100)
    generate_data_file('./data/problem6_c.dat', 2, lambda x, y: x**2 + 3*x*y - 7*y + 1, -1, 1, num_of_points=100)
    generate_data_file('./data/problem6_d.dat', 2, lambda x, y: x**2 + 3*x*y - 7*y + 1, -1000, 1000, num_of_points=100)


    # dodatkowe I
    generate_data_file('./data/problem7_a.dat', 1, lambda x: (1/np.sqrt(2*3.1415)) * np.exp(-(x*x)/2), -4, 4, num_of_points=100)
    generate_data_file('./data/problem7_b.dat', 1, lambda x: (1/np.sqrt(2*3.1415)) * np.exp(-(x*x)/2), 0, 20, num_of_points=100)


    # dodatkowe II
    generate_data_file('./data/problem8_a.dat', 1, lambda x: np.sin(x) + np.cos(x), 0, 7, num_of_points=100)
    generate_data_file('./data/problem8_b.dat', 1, lambda x: 5*x**3 - 2*x**2 + 3*x - 17, -10, 10, num_of_points=100)
    generate_data_file('./data/problem8_c.dat', 1, lambda x: 2* np.log(x+1), 0, 4, num_of_points=100)

    # 9, 2.1
    generate_data_file('./data/problem9_a.dat', 1, lambda x: np.sin(x) + np.cos(x), -3.14, 3.14, num_of_points=100)
    generate_data_file('./data/problem9_b.dat', 1, lambda x: np.sin(x) + np.cos(x), 0, 7, num_of_points=100)
    generate_data_file('./data/problem9_c.dat', 1, lambda x: np.sin(x) + np.cos(x), 0, 100, num_of_points=100)
    generate_data_file('./data/problem9_d.dat', 1, lambda x: np.sin(x) + np.cos(x), -100, 100, num_of_points=100)

    # 10, 2.2
    generate_data_file('./data/problem10_a.dat', 1, lambda x: np.sin(x + 3.141592 / 2), 0, 3.14*2, num_of_points=100)

    # 11, 2.3
    generate_data_file('./data/problem11_a.dat', 1, lambda x: np.tan(2*x + 1), -3.14/2, 3.14/2, num_of_points=100)
