import openpyxl
from xlrd import open_workbook
from xlutils.copy import copy
import os
# 追加方式写入数据
# 读取excel文件，获取workbook对象
# wb = openpyxl.load_workbook("D:/用过的数据.xlsx")
file_dir = r"D:\第二部分标记数据\股骨滑车发育不良\原图\最近端第一张软骨滑车轴位像"
def getFlist(path):
    for root, dirs, files in os.walk(path):
        print('root_dir:', root)  #当前路径
        print('sub_dirs:', dirs)   #子文件夹
        print('files:', files)     #文件名称，返回list类型
    return files
file_name = getFlist(file_dir)

def write_to_excel(list, goal):
    r_xls = open_workbook(goal)  # 读取excel文件
    row = r_xls.sheets()[0].nrows  # 获取已有的行数
    excel = copy(r_xls)  # 将xlrd的对象转化为xlwt的对象
    worksheet = excel.get_sheet(0)  # 获取要操作的sheet
    # 对excel表追加一行内容
    for name in list:
        worksheet.write(row, 0, name[:9])  # 括号内分别为行数、列数、内容
        row += 1
    # 增加第二列内容
    excel.save(goal)  # 保存并覆盖文件

write_to_excel(file_name, "D:\用过的数据.xlsx")