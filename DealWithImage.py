import re
import sys
import base64
import os

# 正则表达式
p1 = re.compile(r'[{](.*?)[}]', re.S)
p2 = re.compile(r"['](.*?)[']", re.S)
p3 = re.compile(r'[#](.*?)[,]', re.S)

# 创建result目录
# 获得当前路径
curPath = os.getcwd()
tempPath = 'result'
# os.path.sep 斜杠/
targetPath = curPath + os.path.sep + tempPath
if not os.path.exists(targetPath):
	os.makedirs(targetPath)

# 遍历result_txt内所有文件
path = "result_txt"
file_names = []
for root, dirs, files in os.walk(path):
	for file in files:
		if '.txt' in file:
			file_names.append(file)

# 将txt文件转成svg
for file_name in file_names:
	word_name = file_name[:1]
	print(word_name)
	# 创建子目录
	targetChildrenPath = targetPath + os.path.sep + word_name
	if not os.path.exists(targetChildrenPath):
		os.makedirs(targetChildrenPath)

	# 读取文件
	f = open('result_txt/' + file_name, 'r')
	lines = f.readlines()
	file_number = []

	# svg文件
	for line in lines:
		if len(line) >= 25:
			# 获取css内容
			list_word = re.findall(p1, line)

			# 获取文字编号
			number = re.findall(p3, line)
			for num in number:
				if len(num) < 50:
					file_number.append(num)

			for i in range(len(file_number)):
				# 甲骨文
				if 'J' in file_number[i]:
					# 创建子文件夹
					targetGrandchildrenPath = targetChildrenPath + os.path.sep + '甲骨文'
					if not os.path.exists(targetGrandchildrenPath):
						os.makedirs(targetGrandchildrenPath)

					# 创建svg子文件夹
					targetGrandgrandchildrenPath = targetGrandchildrenPath + os.path.sep + 'svg'
					if not os.path.exists(targetGrandgrandchildrenPath):
						os.makedirs(targetGrandgrandchildrenPath)

					# 创建png子文件夹
					targetGrandgrandchildrenPath = targetGrandchildrenPath + os.path.sep + 'png'
					if not os.path.exists(targetGrandgrandchildrenPath):
						os.makedirs(targetGrandgrandchildrenPath)
					
					# 获取{}中间内容
					content = re.findall(p2, list_word[i])
					base64__ = "".join(str(x) for x in content)[26:]
					# 解码base64
					svg = base64.b64decode(base64__)

					file_word_name = 'result/' + word_name + '/' + '甲骨文' + '/svg/' + file_number[i] + '.svg'
					with open(file_word_name, 'w', encoding='utf-8', newline='') as c:
						c.write(str(svg, encoding='utf-8'))

				# 金文
				elif 'B' in file_number[i]:
					# 创建子文件夹
					targetGrandchildrenPath = targetChildrenPath + os.path.sep + '金文'
					if not os.path.exists(targetGrandchildrenPath):
						os.makedirs(targetGrandchildrenPath)

					# 创建svg子文件夹
					targetGrandgrandchildrenPath = targetGrandchildrenPath + os.path.sep + 'svg'
					if not os.path.exists(targetGrandgrandchildrenPath):
						os.makedirs(targetGrandgrandchildrenPath)

					# 创建png子文件夹
					targetGrandgrandchildrenPath = targetGrandchildrenPath + os.path.sep + 'png'
					if not os.path.exists(targetGrandgrandchildrenPath):
						os.makedirs(targetGrandgrandchildrenPath)

					# 获取{}中间内容
					content = re.findall(p2, list_word[i])
					base64__ = "".join(str(x) for x in content)[26:]
					# 解码base64
					svg = base64.b64decode(base64__)

					file_word_name = 'result/' + word_name + '/' + '金文' + '/svg/' + file_number[i] + '.svg'
					with open(file_word_name, 'w', encoding='utf-8', newline='') as c:
						c.write(str(svg, encoding='utf-8'))

				# 说文解字的篆字
				elif 'S' in file_number[i]:
					# 创建子文件夹
					targetGrandchildrenPath = targetChildrenPath + os.path.sep + '说文解字的篆字'
					if not os.path.exists(targetGrandchildrenPath):
						os.makedirs(targetGrandchildrenPath)

					# 创建svg子文件夹
					targetGrandgrandchildrenPath = targetGrandchildrenPath + os.path.sep + 'svg'
					if not os.path.exists(targetGrandgrandchildrenPath):
						os.makedirs(targetGrandgrandchildrenPath)

					# 创建png子文件夹
					targetGrandgrandchildrenPath = targetGrandchildrenPath + os.path.sep + 'png'
					if not os.path.exists(targetGrandgrandchildrenPath):
						os.makedirs(targetGrandgrandchildrenPath)
					
					# 获取{}中间内容
					content = re.findall(p2, list_word[i])
					base64__ = "".join(str(x) for x in content)[26:]
					# 解码base64
					svg = base64.b64decode(base64__)

					file_word_name = 'result/' + word_name + '/' + '说文解字的篆字' + '/svg/' + file_number[i] + '.svg'
					with open(file_word_name, 'w', encoding='utf-8', newline='') as c:
						c.write(str(svg, encoding='utf-8'))

				# 六书通
				elif 'L' in file_number[i]:
					# 创建子文件夹
					targetGrandchildrenPath = targetChildrenPath + os.path.sep + '六书通'
					if not os.path.exists(targetGrandchildrenPath):
						os.makedirs(targetGrandchildrenPath)

					# 创建svg子文件夹
					targetGrandgrandchildrenPath = targetGrandchildrenPath + os.path.sep + 'svg'
					if not os.path.exists(targetGrandgrandchildrenPath):
						os.makedirs(targetGrandgrandchildrenPath)

					# 创建png子文件夹
					targetGrandgrandchildrenPath = targetGrandchildrenPath + os.path.sep + 'png'
					if not os.path.exists(targetGrandgrandchildrenPath):
						os.makedirs(targetGrandgrandchildrenPath)
					
					# 获取{}中间内容
					content = re.findall(p2, list_word[i])
					base64__ = "".join(str(x) for x in content)[26:]
					# 解码base64
					svg = base64.b64decode(base64__)

					file_word_name = 'result/' + word_name + '/' + '六书通' + '/svg/' + file_number[i] + '.svg'
					with open(file_word_name, 'w', encoding='utf-8', newline='') as c:
						c.write(str(svg, encoding='utf-8'))

	f.close()
