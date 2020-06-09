import os
import shutil
from PIL import Image

path = 'D:\\project\\data\\'
for filename in os.listdir(path):
	if os.path.exists(path + filename + '\\甲骨文\\png\\'):
		for jinfile in os.listdir(path + filename + '\\甲骨文\\png\\'):
			img_path = path + filename + '\\甲骨文\\png\\' + jinfile
			out_path = path + filename + '\\甲骨文\\png\\'
			im = Image.open(img_path)
			out = im.resize((96, 96))
			x, y = out.size
			try:
				p = Image.new('RGB', [96, 96], (255, 255, 255))
				p.paste(out, (0, 0, x, y), out)
				p.save(out_path + jinfile[0:-4] + '.jpg')
				os.remove(img_path)
			except:
				print(img_path)
	if os.path.exists(path + filename + '\\说文解字的篆字\\png\\'):
		for jinfile in os.listdir(path + filename + '\\说文解字的篆字\\png\\'):
			img_path = path + filename + '\\说文解字的篆字\\png\\' + jinfile
			out_path = path + filename + '\\说文解字的篆字\\png\\'
			im = Image.open(img_path)
			out = im.resize((96, 96))
			x, y = out.size
			try:
				p = Image.new('RGB', [96, 96], (255, 255, 255))
				p.paste(out, (0, 0, x, y), out)
				p.save(out_path + jinfile[0:-4] + '.jpg')
				os.remove(img_path)
			except:
				print(img_path)
