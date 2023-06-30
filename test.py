import datetime

from chinese_calendar import is_workday
data = datetime.date(2023, 6, 1)
if is_workday(data):
  print("是工作日")
else:
  print("是节假日")