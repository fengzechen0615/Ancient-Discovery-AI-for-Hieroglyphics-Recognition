from PIL import Image
import os


class_path='D:\\项目\\毕设\\images-金文\\'
out_path='D:\\项目\\毕设\\images_金文\\'
for img_name in os.listdir(class_path): 
	if img_name != '.DS_Store':
		img_path = class_path+img_name 
		
		im = Image.open(img_path)
		out = im.resize((96,96))
		x, y = out.size
		try:
			p = Image.new('RGB', [96,96], (255,255,255))
			p.paste(out, (0,0,x,y), out)
			p.save(out_path + img_name[0:-4] + '.jpg')
			os.remove(class_path + img_name)
		except:
			print(img_name)