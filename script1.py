import os
import shutil

path = '/Users/glh/Desktop/甲骨文收集/甲骨文-金文/'

for filename in os.listdir(path):
	if(filename != '.DS_Store'):
		if os.path.exists(path + filename + '/金文') == False:
			print(path + filename)
			shutil.rmtree(path+filename)
		