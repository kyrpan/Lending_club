import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import math
from sklearn.metrics import mean_squared_error
sns.set()

def plots(df):
	#1: line plot of income and loan amount
	income = pd.DataFrame().assign(annual_income=df['annual_income'], loan_amount=df['loan_amount'])
	income.plot.line()
	plt.title('Income vs Loan amount')
	plt.show()

	#2: pie plot of the number of years in the job
	emp_length = df['emp_length'].dropna()
	emp_length_counts = emp_length.value_counts()
	colors = sns.color_palette('pastel')[0:10]
	plt.pie(emp_length_counts, labels=emp_length_counts.index, colors=colors, autopct='%.0f%%')
	plt.title('Number of years in the job')
	plt.show()

	#3: bar plot of the loan purpose
	loan_purpose_counts = df['loan_purpose'].value_counts()
	loan_purpose_counts.plot(x='lab', y='val', rot=90, kind='bar')
	plt.tight_layout()
	plt.title('Loan purpose')
	plt.show()

	#4: horizontal bar plot of the states of residence
	state_counts = df['state'].value_counts()
	state_counts.plot(x='lab', y='val', kind='barh', fontsize=5)
	plt.title('State of residence')
	plt.show()

	#5: scatter plot of the homeownership
	homeownership_counts = df['homeownership'].value_counts()
	homeownership_counts.plot(x='lab', y='val', style='.', fontsize=7)
	plt.title('Type of homeownership')
	plt.show()

#one hot encoding of categorical values
def categorical_encoding(data, col):
	encoder = OneHotEncoder(handle_unknown='ignore')
	encoder_df = pd.DataFrame(encoder.fit_transform(data[[col]]).toarray())	

	#set unique labels for the new columns
	cols_count = len(encoder_df.axes[1])
	encoder_columns = []
	for cols in range(cols_count):
		encoder_columns.append(col+'_'+str(cols))
	encoder_df.columns = encoder_columns

	final_data = data.join(encoder_df.set_index(data.index))
	final_data = final_data.drop(columns=col)

	return final_data

def preprocessing(df):
	#keep the columns that are decisive for the interest rate (plus the interest rate)
	data = df.iloc[:,1:43]

	#drop columns that have a lot of NA values or a lot of categorical values
	data = data.drop(columns=['state','earliest_credit_line','annual_income_joint','verification_income_joint','debt_to_income_joint',
		'months_since_last_delinq', 'months_since_90d_late'])

	#drop NA rows
	for column in data:
		data = data[data[column].notna()]

	data = categorical_encoding(data,'homeownership')
	data = categorical_encoding(data,'verified_income')
	data = categorical_encoding(data,'loan_purpose')
	data = categorical_encoding(data,'application_type')

	return data

def plot_results(y_test, y_pred, title):
	df1 = pd.DataFrame({'Actual': y_test[:20].values.flatten(), 'Predicted': y_pred[:20].flatten()}) #a slice of the set
	df1.plot(kind='bar', rot=0)
	plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
	plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
	plt.title(title)
	plt.show()

def main():
	#load csv file
	df = pd.read_csv('loans_full_schema.csv')

	plots(df)	

	#preprocessing of the data
	df = preprocessing(df)

	#split train and test sets
	y = df['interest_rate']
	X = df.drop(columns=['interest_rate'])
	X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.2, random_state = 0)

	#scaling of the data
	sc_X = StandardScaler()
	X_train = sc_X.fit_transform(X_train)
	X_test = sc_X.transform(X_test)

	#Linear Regression
	regressor = LinearRegression()
	regressor.fit(X_train, y_train)
	y_pred = regressor.predict(X_test)
	rmse = math.sqrt(mean_squared_error(y_test,y_pred))
	print (rmse) #Around 0.001 is great, 1.0 - 2.0 means you should tune your model, greater than that means if tuning doesn't work, try another model.

	plot_results(y_test, y_pred, 'LinearRegression')

	#SVR
	regressor = SVR(kernel = 'rbf')
	regressor.fit(X_train, y_train)
	y_pred = regressor.predict(X_test)
	rmse = math.sqrt(mean_squared_error(y_test,y_pred))
	print (rmse)

	plot_results(y_test, y_pred, 'SVR')


if __name__ == '__main__':
	main()
