#  Included: HK Housing price, HK GDP, HK Inflation, HSI, HK Unemployment,
#  Included: Chinese housing price, Chinese GDP, Chinese Inflation, CSI300,
#  Included: sp500, US interest, USD:RMB
#  Not include: chi-hk, Gold, Chinese Deposit, wage, Chinese interest
import quandl
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
import bs4 as bs
from sklearn import svm, preprocessing, cross_validation

# HSI Volume, HSI
# Sourced from: Yahoo
def hsi_data():
    df = quandl.get("YAHOO/INDEX_HSI", authtoken="please use your authtoken from quandl")
    df.rename(columns={'Adjusted Close':'HSI', 'Volume':'HSI Volume'}, inplace=True)
    df.drop(['Open', 'High', 'Low', 'Close'], 1, inplace=True)
    return df
# print hsi_data()

# CSI300 Volume, CSI300
# Sourced from: Google
def csi300_data():
    df = quandl.get("DY7/399300", authtoken="please use your authtoken from quandl")
    df.rename(columns={'Close':'CSI300', 'Volume':'CSI300 Volume'}, inplace=True)
    df.drop(['Open', 'High', 'Low', 'Turnover_Value', 'Turnover_Volume'], 1, inplace=True)
    return df
#print csi300_data()

# SP500
# Sourced from: Yahoo
def sp500_data():
    start = datetime.datetime(2000, 1, 1)
    end = datetime.date.today()
    df = web.DataReader("^GSPC", "yahoo", start, end)
    df.drop(['Open', 'High', 'Low', 'Volume', 'Close'], 1, inplace=True)
    df.rename(columns={'Adj Close':'SP500'}, inplace=True)
    return df
# print sp500_data()

# HK GDP
# Hong Kong SAR GDP at Current Prices, USD Billions
# Sourced from: IMF
def hk_gdp_data():
    df = quandl.get("ODA/HKG_NGDPD", authtoken="please use your authtoken from quandl")
    df.rename(columns={'Value':'HK GDP'}, inplace=True)
    return df
# print hk_gdp_data()

# HK Unemployment
# Hong Kong SAR Unemployment Rate, % of Total Labor Force
# Sourced from: IMF
def hk_unem_data():
    df = quandl.get("ODA/HKG_LUR", authtoken="please use your authtoken from quandl")
    df.rename(columns={'Value':'HK Unemployment'}, inplace=True)
    return df
# print hk_unem_data()

# HK Inflation
# Hong Kong SAR Inflation Index, Average Consumer Prices
# Sourced from: IMF
def hk_inflation_data():
    df = quandl.get("ODA/HKG_PCPI", authtoken="please use your authtoken from quandl")
    df.rename(columns={'Value':'HK Inflation'}, inplace=True)
    return df
# print hk_inflation_data()




# CH GDP
# China GDP at Current Prices, USD Billions
# Sourced from: IMF
def ch_gdp_data():
    df = quandl.get("ODA/CHN_NGDPD", authtoken="please use your authtoken from quandl")
    df.rename(columns={'Value':'CH GDP'}, inplace=True)
    return df
# print ch_gdp_data().tail()

# CH Inflation
# China Inflation Index, Average Consumer Prices
# Sourced from: IMF
def ch_inflation_data():
    df = quandl.get("ODA/CHN_PCPI", authtoken="please use your authtoken from quandl")
    df.rename(columns={'Value':'CH Inflation'}, inplace=True)
    return df
# print ch_inflation_data().tail()

# CH Housing
# CSI China Mainland Real Estate Index
# Sourced from: Google
# 2016 latest
def ch_housing_data():
    df = quandl.get("GOOG/SHA_000948", authtoken="please use your authtoken from quandl")
    df.drop(['Open', 'High', 'Low', 'Volume'], 1, inplace=True)
    df.rename(columns={'Close':'CH Housing'}, inplace=True)
    return df
# print ch_housing_data().tail()

# US Interest
# 30 Year
# Sourced from: Yahoo
def us_interest_data():
    start = datetime.datetime(2000, 1, 1)
    end = datetime.date.today()
    df = web.DataReader("^TYX", "yahoo", start, end)
    df.drop(['Open', 'High', 'Low', 'Volume', 'Close'], 1, inplace=True)
    df.rename(columns={'Adj Close':'US Interest'}, inplace=True)
    return df
# print us_interest_data().tail()

# USD to CNY
def usd_to_cny_data():
    df = quandl.get("CUR/CNY", authtoken="please use your authtoken from quandl")
    df.rename(columns={'RATE':'USD to CNY'}, inplace=True)
    df.index.names = ['Date']
    return df
# print usd_to_cny_data().tail()

# CCL HK part1
# Used start date as index
def ccl_p1_data():
    df = pd.read_excel('/path/CCL/CCLhist.xlsx')
    df.drop(['End'], 1, inplace=True)
    df['Date'] = df['Start'].apply(lambda x: x.date())
    df.drop(['Start'], 1, inplace=True)
    df = df.set_index('Date')
    return df
# print ccl_p1_data()

# CCL HK part2
# Used start date as index
def ccl_p2_data():
    resp = open('/path/CCL/CCL2014.htm', 'r')
    soup = bs.BeautifulSoup(resp)
    table = soup.find('table', {'id':'AutoNumber1'})
    d = []
    for row in table.findAll('tr')[1:]:
        dates = row.findAll('td')[0].text.split('-')
        date = dates[0].strip()
        ccl = row.findAll('td')[1].text
        d.append({'Date':str(date), 'CCL':str(ccl)})
    d = d[:-1]
    df = pd.DataFrame(d[::-1])
    df2 = df.set_index('Date')
    return df2
# print ccl_p2_data()

# CCL HK part3
# Used start date as index
def ccl_p3_data():
    resp = open('/path/CCL/CCL2015.htm', 'r')
    soup = bs.BeautifulSoup(resp)
    table = soup.find('table', {'id':'AutoNumber1'})
    d = []
    for row in table.findAll('tr')[1:]:
        dates = row.findAll('td')[0].text.split('-')
        date = dates[0].strip()
        ccl = row.findAll('td')[1].text
        d.append({'Date':str(date), 'CCL':str(ccl)})
    df = pd.DataFrame(d[::-1])
    df2 = df.set_index('Date')
    return df2
# print ccl_p3_data()

# CCL HK part4
# Used start date as index
def ccl_p4_data():
    resp = open('/path/CCL/CCL2016.htm', 'r')
    soup = bs.BeautifulSoup(resp)
    table = soup.find('table', {'id':'AutoNumber1'})
    d = []
    for row in table.findAll('tr')[1:]:
        dates = row.findAll('td')[0].text.split('-')
        date = dates[0].strip()
        ccl = row.findAll('td')[1].text
        d.append({'Date':str(date), 'CCL':str(ccl)})
    df = pd.DataFrame(d[::-1])
    df2 = df.set_index('Date')
    return df2
# print ccl_p4_data()

# CCL HK part5
# Used start date as index
# 2017 Jan 1 to 2017 today
# http://202.72.14.52/p2/cci/SearchHistory.aspx
def ccl_p5_data():
    resp = open('/path/CCL/CCL2017.htm', 'r')
    soup = bs.BeautifulSoup(resp)
    table = soup.find('table', {'id':'AutoNumber1'})
    d = []
    for row in table.findAll('tr')[1:]:
        dates = row.findAll('td')[0].text.split('-')
        date = dates[0].strip()
        ccl = row.findAll('td')[1].text
        d.append({'Date':str(date), 'CCL':str(ccl)})
    df = pd.DataFrame(d[::-1])
    df2 = df.set_index('Date')
    return df2
# print ccl_p5_data()

def ccl_compile():
    p1 = ccl_p1_data()
    p2 = ccl_p2_data()
    p3 = ccl_p3_data()
    p4 = ccl_p4_data()
    p5 = ccl_p5_data()
    ccl = p1.append([p2, p3, p4, p5])
    return ccl
#print ccl_compile()

data_list = ['hsi_data', 'csi300_data', 'sp500_data', 'hk_gdp_data',
'hk_unem_data', 'hk_inflation_data', 'ch_gdp_data','ch_inflation_data',
'ch_housing_data', 'us_interest_data', 'usd_to_cny_data', 'ccl_compile']

# Need to download CCL2017 manually first
def download_all_data():
    path = r'/path/hk_data/'
    p1 = hsi_data()
    p1.to_csv(path + 'hsi_data.csv')
    p2 = csi300_data()
    p2.to_csv(path + 'csi300_data.csv')
    p3 = sp500_data()
    p3.to_csv(path + 'sp500_data.csv')
    p4 = hk_gdp_data()
    p4.to_csv(path + 'hk_gdp_data.csv')
    p5 = hk_unem_data()
    p5.to_csv(path + 'hk_unem_data.csv')
    p6 = hk_inflation_data()
    p6.to_csv(path + 'hk_inflation_data.csv')
    p7 = ch_gdp_data()
    p7.to_csv(path + 'ch_gdp_data.csv')
    p8 = ch_inflation_data()
    p8.to_csv(path + 'ch_inflation_data.csv')
    p9 = ch_housing_data()
    p9.to_csv(path + 'ch_housing_data.csv')
    p10 = us_interest_data()
    p10.to_csv(path + 'us_interest_data.csv')
    p11 = usd_to_cny_data()
    p11.to_csv(path + 'usd_to_cny_data.csv')
    p12 = ccl_compile()
    p12.to_csv(path + 'ccl_compile.csv')
    print ("Data downloaded.")
# download_all_data()

# Format all data
def format_data():
    path = r'/path/hk_data/'
    # HSI Volume, HSI
    p1 = pd.read_csv(path + ('hsi_data.csv'))
    p1['Date'] = p1['Date'].apply(pd.to_datetime)
    p1.set_index('Date', inplace=True)
    p1 = p1.resample('W').mean()

    # CSI300
    p2 = pd.read_csv(path + ('csi300_data.csv'))
    p2['Date'] = p2['Date'].apply(pd.to_datetime)
    p2.set_index('Date', inplace=True)
    p2 = p2.resample('W').mean()

    # SP500
    p3 = pd.read_csv(path + ('sp500_data.csv'))
    p3['Date'] = p3['Date'].apply(pd.to_datetime)
    p3.set_index('Date', inplace=True)
    p3 = p3.resample('W').mean()

    # HK GDP
    p4 = pd.read_csv(path + ('hk_gdp_data.csv'))
    p4['Date'] = p4['Date'].apply(pd.to_datetime)
    p4.set_index('Date', inplace=True)
    p4 = p4.resample('W').mean()
    p4.fillna(method='ffill', inplace=True)

    # HK Unemployment
    p5 = pd.read_csv(path + ('hk_unem_data.csv'))
    p5['Date'] = p5['Date'].apply(pd.to_datetime)
    p5.set_index('Date', inplace=True)
    p5 = p5.resample('W').mean()
    p5.fillna(method='ffill', inplace=True)

    # HK Inflation
    p6 = pd.read_csv(path + ('hk_inflation_data.csv'))
    p6['Date'] = p6['Date'].apply(pd.to_datetime)
    p6.set_index('Date', inplace=True)
    p6 = p6.resample('W').mean()
    p6.fillna(method='ffill', inplace=True)

    # CH GDP
    p7 = pd.read_csv(path + ('ch_gdp_data.csv'))
    p7['Date'] = p7['Date'].apply(pd.to_datetime)
    p7.set_index('Date', inplace=True)
    p7 = p7.resample('W').mean()
    p7.fillna(method='ffill', inplace=True)

    # CH Inflation
    p8 = pd.read_csv(path + ('ch_inflation_data.csv'))
    p8['Date'] = p8['Date'].apply(pd.to_datetime)
    p8.set_index('Date', inplace=True)
    p8 = p8.resample('W').mean()
    p8.fillna(method='ffill', inplace=True)

    # CH Housing
    # NOT USED
    #p9 = pd.read_csv(path + ('ch_housing_data.csv'))
    #p9['Date'] = p9['Date'].apply(pd.to_datetime)
    #p9.set_index('Date', inplace=True)
    #p9 = p9.resample('W').mean()
    #p9.fillna(method='ffill', inplace=True)

    # US Interest
    p10 = pd.read_csv(path + ('us_interest_data.csv'))
    p10['Date'] = p10['Date'].apply(pd.to_datetime)
    p10.set_index('Date', inplace=True)
    p10 = p10.resample('W').mean()


    # USD to CNY
    p11 = pd.read_csv(path + ('usd_to_cny_data.csv'))
    p11['Date'] = p11['Date'].apply(pd.to_datetime)
    p11.set_index('Date', inplace=True)
    p11 = p11.resample('W').mean()

    # CCL
    p12 = pd.read_csv(path + ('ccl_compile.csv'))
    p12['Date'] = p12['Date'].apply(pd.to_datetime)
    p12.set_index('Date', inplace=True)
    p12 = p12.resample('W').mean()

    df = p1.join([p2, p3, p4, p5, p6, p7, p8, p10, p11, p12])
    # CCL (2wks ffill)
    df['CCL'].fillna(method='ffill', inplace=True)
    return df

# print format_data()
def create_labels(cur_hsi, fut_hsi):
    profit_counter=1
    if fut_hsi > cur_hsi and fut_hsi > 0.03:  # if rise 3%
        profit_counter = profit_counter * (fut_hsi)
        return 1
    else:
        return 0
    print profit_counter

def process():
    df = format_data()
    df[['HSI Volume', 'HSI', 'CSI300', 'SP500', 'US Interest', 'USD to CNY', 'CCL']] = df[['HSI Volume', 'HSI', 'CSI300', 'SP500', 'US Interest', 'USD to CNY', 'CCL']].pct_change()
    df['CCL'].replace(to_replace=0, method='ffill')
    # shift future value to current date
    df['HSI_future'] = df['HSI'].shift(-1)
    df.replace([-np.inf, np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df['label'] = list(map(create_labels, df['HSI'], df['HSI_future']))
    # print df.tail(50)
    X = np.array(df.drop(['label', 'HSI_future'], 1)) # 1 = column
    X = preprocessing.scale(X)
    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

#print process()
accuracies = []
for j in range(10):
    number = process()
    accuracies.append(number)
print 'Tests: ', accuracies, 'Mean Accuracy', sum(accuracies) / len(accuracies)
