import sqlite3
import pandas as pd

titanic_df = pd.read_csv('Data_files/titanic_train.csv')
columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
           'Central Air', 'Total Bsmt SF', 'SalePrice','Fireplaces', 'Exter Qual']
housing_df = pd.read_csv('Data_files/AmesHousing.txt', sep = '\t', usecols=columns)
movie_df = pd.read_csv('Data_files/movie_data.csv')


#create connection to database

try:
    conn = sqlite3.connect("final_project.db")
    titanic_df.to_sql('titanic', conn, if_exists='replace',index=False)
    housing_df.to_sql('housing',conn,if_exists='replace',index=False)
    movie_df.to_sql('movies',conn,if_exists='replace',index=False)

except sqlite3.Error as e:
    print(e)

finally:
    conn.close()

