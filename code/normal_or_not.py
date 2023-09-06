fa = open(r"D:\code\Knee\data\knee\group\good.txt")
txt1 = fa.read()
fa.close()

fb = open(r"D:\code\Knee\data\knee\group\cv\3\test.txt")
txt2 = fb.read()
fb.close()

line1 = txt1.splitlines()
line2 = txt2.splitlines()

overlapfile = open(r"D:\code\Knee\data\knee\group\result_overlap.txt",'w')
for i in line1:
	# 查看a文件中的元素是否在b文件中存在
	if i in line2:
		overlapfile.writelines(i+'\n') #将在b文件中也存在的元素保存到result_overlap.txt文件中，\n为换行符。这样在最后输出的文件中各个元素也是按照换行符分割。
print("检查结束")
