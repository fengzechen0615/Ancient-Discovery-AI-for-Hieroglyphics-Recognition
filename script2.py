import os
import shutil

path = 'D:\\project\\result\\'
count1 = 0
count2 = 0
count3 = 0
count11 = 0
count22 = 0

for filename in os.listdir(path):
	if os.path.exists(path+filename+"\\甲骨文"):
		count1 += 1
		for char in os.listdir(path+filename+"\\甲骨文\\png\\"):
			count11 += 1
	if os.path.exists(path+filename+"\\金文"):
		count2 += 1
		for char in os.listdir(path+filename+"\\金文\\png\\"):
			count22 += 1
	if os.path.exists(path+filename+"\\说文解字的篆字"):
		count3 += 1

print(count1)
print(count2)
print(count3)
print(count11)
print(count22)
