import xlrd


with open((r'D:/code/Knee/code/result/model_3cm.txt'), "r") as f1:
    result_3cm = [item.replace('\n', '') for item in f1.readlines()]
print(result_3cm[0])
# workbook = xlrd.open_workbook("D:/code/Knee/data/knee/距离角度误差.xls")
# # 获取原表格第一个sheet的名字
# all_sheet = workbook.sheet_names()
# first_sheet = workbook.sheet_by_name(all_sheet[0])
# # 获取原表格第一个sheet一写入数据的行数
# rows = first_sheet.nrows

