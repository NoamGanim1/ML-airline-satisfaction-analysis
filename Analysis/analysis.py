import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a DataFrame
df = pd.read_csv(r"training-data.csv")

print(df.head())

pd.set_option('future.no_silent_downcasting', True)


#---------------------------------------Function------------------------------------------------
# Function to plot histograms for categorical data with percentages
def plot_histogram(data, title, xlabel, ylabel, labels, colors):
    counts = data.value_counts(normalize=True) * 100
    ax = counts.plot(kind='bar', color=colors)
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(labels)), labels, rotation='horizontal')
    for i, v in enumerate(counts.values):
        ax.text(i - 0.05, v + 1, f"{v:.1f}%", fontweight='bold')
    plt.show()

# Function to plot histograms for numerical data with percentages
def plot_numerical_histogram(data, bins, title, xlabel, ylabel, color):
    data = data.dropna()  # Drop missing values
    counts, bins_edges = np.histogram(data, bins=bins, density=True)
    counts = counts * np.diff(bins_edges) * 100  # Convert to percentage
    bins_center = (bins_edges[:-1] + bins_edges[1:]) / 2

    plt.bar(bins_center, counts, width=np.diff(bins_edges), color=color, edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# Function to plot histograms for Categorical variables that we will treat as continuous
def plot_bar(data, title, xlabel, ylabel, color):
    counts = data.value_counts(normalize=True) * 100
    ax = counts.sort_index().plot(kind='bar', color=color, edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, counts.max() + 10)  # Adjust ylim for better visibility
    plt.show()

#---------------------------------------Question 2.1 + 2.2------------------------------------------------

# 0 target variable - satisfaction
df_Satisfaction = df['satisfaction'].replace({'satisfied': 0, 'neutral or dissatisfied': 1})
plot_histogram(df_Satisfaction, 'Histogram of Satisfaction', 'Satisfaction', 'Percentage', ['neutral or dissatisfied', 'satisfied'], ['red', 'pink'])

# Categorical variables (1, 2, 3)
# 1 Gender
df_Gender = df['Gender'].replace({'Male': 1, 'Female': 0})
plot_histogram(df_Gender, 'Histogram of Genders', 'Gender', 'Percentage', ['Male', 'Female'], ['pink', 'lightblue'])

# 2 Customer Type
df_Customers = df['Customer Type'].replace({'Loyal Customer': 0, 'disloyal Customer': 1})
plot_histogram(df_Customers, 'Histogram of Customers Type', 'Customer Type', 'Percentage', ['Loyal Customer', 'disloyal Customer'], ['blue', 'red'])

# 3 Type of Travel
df_Type = df['Type of Travel'].replace({'Personal Travel': 0, 'Business travel': 1})
plot_histogram(df_Type, 'Histogram of Type of Travel', 'Type of Travel', 'Percentage', ['Personal Travel', 'Business travel'], ['orange', 'pink'])

# Continuous variables (4, 5, 6, 7, 8, 9, 10)

# 4 Flight Distance histogram with percentages
plot_numerical_histogram(df['Flight Distance'], bins=50, title='Flight Distance Histogram', xlabel='Flight Distance', ylabel='Percentage', color='purple')

# 5 Age histogram with percentages
plot_numerical_histogram(df['Age'], bins=32, title='Age Histogram', xlabel='Age', ylabel='Percentage', color='red')

# 6 Departure Delay in Minutes histogram with percentages
plot_numerical_histogram(df['Departure Delay in Minutes'], bins=122, title='Departure Delay in Minutes Histogram', xlabel='Departure Delay in Minutes', ylabel='Percentage', color='blue')

# 7 Arrival Delay in Minutes histogram with percentages
plot_numerical_histogram(df['Arrival Delay in Minutes'], bins=122, title='Arrival Delay in Minutes Histogram', xlabel='Arrival Delay in Minutes', ylabel='Percentage', color='green')


# Categorical variables that we will treat as continuous (8, 9, 10)

# 8 Inflight wifi service with percentages
plot_bar(df['Inflight wifi service'], 'Histogram of Inflight wifi service', 'Inflight wifi service', 'Percentage', 'purple')

# 9 Food and drink service with percentages
plot_bar(df['Food and drink'], 'Histogram of Food and drink', 'Food and drink', 'Percentage', 'black')

# 10 Seat comfort with percentages
plot_bar(df['Seat comfort'], 'Histogram of Seat comfort', 'Seat comfort', 'Percentage', 'yellow')

#---------------------------------------Question 2.3------------------------------------------------

# List of columns to plot
columns = [
    'Age', 'Flight Distance', 'Plane colors',
    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Seat comfort', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
    'Departure Delay in Minutes', 'Arrival Delay in Minutes']
# corr matrix of columns

corr_matrix = df[columns].corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, square=True)
plt.title('Correlation Matrix')
plt.show()

# expected columns with high correlation
col = ['Cleanliness', 'Food and drink', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

corelation = df[col].corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corelation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, square=True)
plt.title(f'Correlation Matrix for {col}', fontsize=16)  # Adjust title font size
plt.title('Correlation Matrix')
plt.show()

# unexpected columns with high correlation
unexpectedCol = ['Age','Seat comfort','Flight Distance', 'Leg room service']

corelation = df[unexpectedCol].corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corelation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, square=True)
plt.title(f'Correlation Matrix for {unexpectedCol}', fontsize=16)  # Adjust title font size
plt.title('Correlation Matrix')
plt.show()

#---------------------------------------Question 2.5------------------------------------------------

#Change string values to Boolean in the type column satisfaction
df['satisfaction'] = df['satisfaction'].replace({'satisfied': 1, 'neutral or dissatisfied': 0})

# List of columns to plot
columns2 = ['Baggage handling', 'Seat comfort', 'Arrival Delay in Minutes', 'satisfaction']

#Correlation between variables
corr_matrix = df[columns2].corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, square=True)
plt.title('Correlation Matrix')
plt.show()

# List of columns to plot
columns3 = [
    'Age', 'Flight Distance', 'Plane colors',
    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Seat comfort', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
    'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'satisfaction']

#Correlation between variables
corr_matrix = df[columns3].corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, square=True)
plt.title('Correlation Matrix')
plt.show()


#---------------------------------------Question 3.1------------------------------------------------

# new data frame
dfnew = pd.read_csv(r"training-data.csv")

# display the row numbers with more than 5 missing values and show the number of missing values in each row
missing_values = df.isnull().sum(axis=1)
missing_values = missing_values[missing_values > 5]
print(missing_values)

#print the missing_values of all the columns
missing_values = dfnew.isnull().sum()
print(missing_values)

#---------------------------------------Question 4.1 (data preparation)------------------------------------------------
# remove row 5568 and 5602 with missing values
dfnew = dfnew.drop([5568, 5602])

#print the missing_values of all the columns after remove rows
missing_values = dfnew.isnull().sum()
print(missing_values)

# remove columns leg room service and arrival delay in minutes
dfnew = dfnew.drop(columns=['Leg room service', 'Arrival Delay in Minutes'])
print(dfnew.head())

# check how many missing values are in the new dataset
missing_values = dfnew.isnull().sum()
print(missing_values)

#---------------------------------------Question 4.2------------------------------------------------
# Age : change data over 120 to the average
dfnew.loc[dfnew['Age'] > 120, 'Age'] = dfnew['Age'].mean().astype(int)

# Flight Distance : change negative data to the average
dfnew.loc[dfnew['Flight Distance'] < 0, 'Flight Distance'] = dfnew['Flight Distance'].mean().astype(int)

# Gate location : change data over 5 to the most common value
dfnew.loc[dfnew['Gate location'] > 5, 'Gate location'] = dfnew['Gate location'].mode()[0]

#Inflight service : change data under 1 to the most common value
dfnew.loc[dfnew['Inflight service'] < 1, 'Inflight service'] = dfnew['Inflight service'].mode()[0]

#Class:

#Changes the exception fields to Business for histogram
row_index = 8732
dfnew.loc[row_index, 'Class'] ='Business'

#.................Create the histogram Class and Type of Travel...............

# Group data by type of travel
travel_groups = dfnew.groupby('Type of Travel')

# Create the histogram with clear labels and title
fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size for better readability

for name, group in travel_groups:
    ax.hist(group['Class'], label=name, alpha=0.7)  # Adjust alpha for transparency

ax.set_xlabel('Travel Class')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Travel Class by Type of Travel')

# Add legend and grid for clarity
ax.legend()
ax.grid(True)

plt.show()

#Changes the exception fields to Business or Eco
dfnew.loc[(dfnew['Class'] == 'Unknown') & (dfnew['Type of Travel'] == 'Personal Travel'), 'Class'] = 'Eco'
dfnew.loc[(dfnew['Class'] == 'Unknown') & (dfnew['Type of Travel'] == 'Business travel'), 'Class']='Business'


#---------------------------------------Question 5------------------------------------------------

#5.1  Flight Distance histogram
bins = [0, 500, 1000, 2600, dfnew['Flight Distance'].max()]
labels = ['0-500', '500-1000', '1000-2600', '2600+']
# Create a temporary Series for plotting
temp = pd.cut(dfnew['Flight Distance'], bins=bins, labels=labels, include_lowest=True)

# Plot the histogram using the temporary Series
plot_histogram(temp, 'Flight Distance Histogram', 'Flight Distance', 'Percentage', labels, ['red', 'blue', 'green', 'purple'])

# Add a new column for 'Flight Distance Rank 1-4
dfnew['Flight Distance Rank'] = temp.cat.rename_categories({'0-500': 1, '500-1000': 2, '1000-2600': 3, '2600+': 4})


# 5.2 Departure Delay in Minutes histogram
bins = [0, 5, 40, dfnew['Departure Delay in Minutes'].max()]
labels = ['0-5', '5-40', '40+']
# Create a temporary Series for plotting
temp = pd.cut(dfnew['Departure Delay in Minutes'], bins=bins, labels=labels, include_lowest=True)

# Plot the histogram using the temporary Series
plot_histogram(temp, 'Departure Delay in Minutes Histogram', 'Departure Delay in Minutes', 'Percentage', labels, ['red', 'blue', 'green'])

# Add a new column for 'Departure Delay Rank 1-3
dfnew['Departure Delay Rank'] = temp.cat.rename_categories({'0-5': 1, '5-40': 2, '40+': 3})


# 5.3 Create new column 'quality service' using average of On-board service and Inflight service
dfnew['Service quality'] = (dfnew['On-board service'] + dfnew['Inflight service'])/2


#5.4 Create new column 'Baggage service' using average of Baggage handling and Check-in service
dfnew['Baggage service'] = (dfnew['Baggage handling'] + dfnew['Checkin service'])/2
