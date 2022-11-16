# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 16:59:06 2022

@author: cuisa
"""
import math
f = open('C:/Users/cuisa/Desktop/python/TSP_Heuristic.txt')
nCities = int(f.readline())
lines = f.readlines()
f.close()


start_coord = (float(lines[0].split()[1]), float(lines[0].split()[2]))


city_record= [i for i in range(1,nCities+1)]
path_record = [0]
visit = set()
visit.add(0)
curr_line  = 0
total_path = 0
while len(visit) != nCities:
    curr_info = lines.pop(curr_line)
    city_record.pop(curr_line)
    curr_coord = (float(curr_info.split()[1]), float(curr_info.split()[2])) 
    edis_list = []
    for city in lines:
        other_coord=( float(city.split()[1]), float(city.split()[2]))
        e_dis = math.sqrt(  (curr_coord[0]- other_coord[0])**2 + (curr_coord[1] - other_coord[1])**2  )
        edis_list.append([e_dis, city.split()[0] ])
    total_path += sorted(edis_list, key=lambda x:x[0])[0][0]
    visit.add(int(sorted(edis_list, key=lambda x:x[0])[0][1]))
    path_record.append(int(sorted(edis_list, key=lambda x:x[0])[0][1]))
    curr_line =  city_record.index(int(sorted(edis_list, key=lambda x:x[0])[0][1]))
    print(len(visit))


final_coord = (float(lines[0].split()[1]), float(lines[0].split()[2]))
final_path = math.sqrt((start_coord[0]-final_coord[0])**2 + (start_coord[1] - final_coord[1])**2)
total_path += final_path
print(math.floor(total_path))
