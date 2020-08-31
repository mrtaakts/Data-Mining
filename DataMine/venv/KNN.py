import sqlite3
import pandas as pd


conn= sqlite3.connect('dataset.sqlite')

df= pd.read_sql_query("SELECT * FROM Iris", conn)

print(df.head()) # iris db'sindeki ilk 5 veriyi gösteriyorum

