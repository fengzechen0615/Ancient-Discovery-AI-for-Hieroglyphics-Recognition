import re
import os

# 遍历result_txt内所有文件
path = "result_txt"
file_names = []
for root, dirs, files in os.walk(path):
	for file in files:
		if '.txt' in file:
			file_names.append(file)

n = 0

for file_name in file_names:
	# 读取文件
	f = open('result_txt/' + file_name, 'r')
	lines = f.readlines()
	file_size = os.path.getsize('result_txt/' + file_name)

	if file_size == 32:
		n = n + 1
		print('remove: ' + file_name)
		os.remove('result_txt/' + file_name)
	else:
		for line in lines:
			if len(line) >= 25:
				if "#J" not in line:
					print('remove: ' + file_name)
					os.remove('result_txt/' + file_name)
					n = n + 1

print('Totel number of files are: ' + str(len(file_names)))
print("Remove " + str(n) + " files!")

