import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime as dt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


data = pd.read_csv('data.csv')


# data.head()

total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)*100
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

data.shape
data = data.dropna()
data.shape

fig_size = plt.rcParams["figure.figsize"]
print(f"Current size : {fig_size}")
fig_size[0], fig_size[1] = 15, 8
plt.rcParams['figure.figsize'] = fig_size
fig_size = plt.rcParams["figure.figsize"]
print(f"New size : {fig_size}")

category = 'Close'

data_stock = data
data_stock.head()

X = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in data_stock['Date']]
y = data_stock['Close']

# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=25)) #x axis tick every 60 days
# plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(2500)) # sets y axis tick spacing to 100
#
# plt.plot(X, y)
# plt.grid(True) #turns on axis grid
# plt.ylim(0) #sets the y axis min to zero
# plt.xticks(rotation=35, fontsize=10)
# plt.title("Samsung Stock analysis") #prints the title on the top
# plt.ylabel(f'Stock Price For {category}') #labels y axis
# plt.xlabel('Date') #labels x axis
# plt.show()

# For specific time frame
# startdate = ('2019-01-31')
# enddate = ('2019-10-16')
#
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=25)) #x axis tick every 60 days
# plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(2500)) # sets y axis tick spacing to 100
#
# plt.plot(X, y)
# plt.grid(True)
# plt.xlim(startdate, enddate)
# plt.ylim(0)
# plt.xticks(rotation=35, fontsize=10)
# plt.title(f"Stock price from {startdate} to {enddate}")
#
# plt.ylabel(f'Stock Price For {category}') #labels y axis
# plt.xlabel('Date') #labels x axis
# plt.show()

# Weekday based trend
# Monday - 0, Sunday - 6

# week_day = {
#     0 : 'Monday',
#     1 : 'Tuesday',
#     2 : 'Wednesday',
#     3 : 'Thursday',
#     4 : 'Friday',
#     5 : 'Saturday',
#     6 : 'Sunday',
# }
import calendar

week_days_integer = [dt.datetime.strptime(d, "%Y-%m-%d").date().weekday() for d in data_stock['Date']]
week_days = [calendar.day_name[day] for day in week_days_integer]

my_day = "Thursday"
data["week_day"] = week_days

data_stock_monday = data[data['week_day'] == my_day]

X_0 = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in data_stock_monday['Date']]
y_0 = data_stock_monday['Close']

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30)) #x axis tick every 60 days
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(2500)) # sets y axis tick spacing to 100

plt.plot(X_0, y_0)
plt.grid(True) #turns on axis grid
plt.ylim(0) #sets the y axis min to zero
plt.xticks(rotation=35, fontsize=10)
plt.title("Samsung Stock analysis") #prints the title on the top
plt.ylabel(f'Stock Price For {category} on {my_day}') #labels y axis
plt.xlabel('Date') #labels x axis
plt.show()
