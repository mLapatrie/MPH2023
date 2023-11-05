import random

list_71 = []
my_nested = []
for i in range(0,71):
    list_71.append(0)

mod_list = [14,15,24,25,50,51,56,57,64,65,66,67]

for nb in mod_list:
    list_71[nb] = 1

for i in range(0,100):
    list_loop = list_71.copy()
    for i in range(len(list_loop)):
        list_loop[i] += random.random()
    for nb in mod_list:
        list_loop[nb] = 1
    my_nested.append(list_loop)

print(my_nested)
