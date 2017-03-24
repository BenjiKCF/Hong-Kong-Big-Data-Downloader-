# Hong Kong Big Data Downloader 
A data downloader for Hong Kong data. This includes HK Housing price (Centa-City Leading Index), HK GDP, HK Inflation, HSI, HK Unemployment, Chinese housing price, Chinese GDP, Chinese Inflation, CSI300, SP500, US interest, USD:RMB
The data are joinned together into a single DataFrame to facilitate future data analysis.
This version will use Quandl and Pandas datareader from yahoo and google to download all the economic data.
Centa-City leading index will be downloaded on  http://202.72.14.52/p2/cci/SearchHistory.aspx and parsed the htm file with beautiful soup
I will also upload my downloaded data to here for the ease of use.
I have also applied a machine learning algorithm in the end to demonstrate how to use the data.
