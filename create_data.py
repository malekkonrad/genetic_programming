import os
import numpy as np
import matplotlib.pyplot as plt
from typing import *

def generate_data():
    
    PATH = './data'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    def generate_data_file(file_name: str, var_num: int, formula: callable, domain_start: float, domain_end: float, num_of_points: Optional[int] = 100):
        if var_num == 1:
            points = np.linspace(domain_start, domain_end, num=num_of_points)
            y_ = formula(points)
            
            plt.plot(points, y_)

            with open(file_name, 'w') as file:
                file.write(f'{var_num} 100 {domain_start} {domain_end} {num_of_points}\n')
                for x in points:
                    y = formula(x)
                    file.write(f'{x} {y}\n')

        if var_num == 2:

            x = np.linspace(domain_start, domain_end, num=num_of_points)
            y = np.linspace(domain_start+1, domain_end+1, num=num_of_points)
            z = formula(x, y)

            fig = plt.figure()
            ax = plt.axes(projection='3d')

            ax.plot3D(x, y, z, 'green')
            ax.set_title('3D Line Plot')
            plt.show()

            with open(file_name, 'w') as file:
                file.write(f'{var_num} 100 {domain_start} {domain_end} {num_of_points}\n')
                for x_, z_ in zip(x, z):
                    file.write(f'{x_} {x_} {z_}\n')

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
    generate_data_file('./data/problem4_a.dat', 2, lambda x, y: x + 2*y , 0, 1, num_of_points=10000)
    generate_data_file('./data/problem4_b.dat', 2, lambda x, y: x + 2*y , -10, 10, num_of_points=10000)
    generate_data_file('./data/problem4_c.dat', 2, lambda x, y: x + 2*y , 0, 100, num_of_points=10000)
    generate_data_file('./data/problem4_d.dat', 2, lambda x, y: x + 2*y , -1000, 1000, num_of_points=10000)

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