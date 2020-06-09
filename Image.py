# -*- coding: UTF-8 -*-

import requests
import csv
import time
import socket
import http.client
import random
from bs4 import BeautifulSoup
import urllib.parse
from selenium import webdriver
from urllib import parse
import os

URL = 'https://hanziyuan.net/#home'

curPath = os.getcwd()
tempPath = 'result_txt'
# os.path.sep 斜杠/
targetPath = curPath + os.path.sep + tempPath
if not os.path.exists(targetPath):
	os.makedirs(targetPath)

# 读取7935个文字
words = []
word_list_file = open('7935.txt', 'r', encoding='UTF-8-sig')
lines = word_list_file.readlines()
for line in lines:
	words.append(line[0:1])

def parse_html(brower):
	final = []
	bs = BeautifulSoup(brower.page_source, 'html.parser')
	
	word_1 = bs.find('div', {'id': 'etymologyResult'})
	
	# 获得css
	word_3 = word_1.find('style', {'type': 'text/css'})

	final.append(word_3)

	return final

def get_data(number):

	try:
		# 打开Google浏览器，请求链接，打开链接
		# path为google驱动完整路径
		# http://chromedriver.storage.googleapis.com/index.html 找到自己的chrome版本驱动下载
		path = '/Users/glh/Desktop/甲骨文收集/chromedriver'
		brower = webdriver.Chrome(executable_path=path)
		# 获得网页内容
		brower.get(URL)
			
		result = parse_html(brower)
		
		# file_name = "result_txt/" + '车.txt'
		# print('车')
		# with open(file_name, 'w', encoding='utf-8', newline='') as f:
		# 	f.write(" ".join(str(x) for x in result))

		time.sleep(10)
		
		for i in range(1000):
			
			# 点击随机按钮
			# element_click = brower.find_element_by_id("etymologyRandomButton")
			# element_click.click()

			# 输入文字 进行搜索
			brower.find_element_by_id("etymologySearchChar").clear()
			brower.find_element_by_id("etymologySearchChar").send_keys(words[number])
			brower.find_element_by_id("etymologySearchButton").click()

			time.sleep(10)

			# 从url获得字名
			url = brower.current_url
			x = '#'
			y = url[url.find(x) + 1:] 
			word_name = parse.unquote(y)
			print(number + 1)
			print(word_name)
			
			result = parse_html(brower)
			
			# 生成相应文件
			# 路径默认根路径
			file_name = "result_txt/" + word_name + '.txt'
			with open(file_name, 'w', encoding='utf-8', newline='') as f:
				f.write(" ".join(str(x) for x in result))
			
			# 滚动至页面顶端
			js = "var q=document.documentElement.scrollTop=0"
			brower.execute_script(js)
			
			# 随机time sleep
			time.sleep(random.choice(range(30, 40)))

			number = number + 1

		# 删除cookie，退出浏览器，重新开始
		brower.delete_all_cookies()
		brower.quit()
		print('New Loop!!!')
		get_data(number)

	except:
		print('Error, restart!!!')
		brower.delete_all_cookies()
		brower.quit()
		get_data(number)



if __name__ == '__main__':
	number = 0
	get_data(4190)

