import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
train_data = pd.read_csv('/Users/jared/Documents/Code/Projects/Mini-Datascience-Projects/TitanicData/train.csv')
test_data = pd.read_csv('/Users/jared/Documents/Code/Projects/Mini-Datascience-Projects/TitanicData/test.csv')

# Combine the datasets for analysis
combined_data = pd.concat([train_data, test_data], sort=False)

# Analyze the survival rate by gender
gender_survival = combined_data.groupby('Sex')['Survived'].mean()
print("Survival rate by gender:")
print(gender_survival)

# Visualize the survival rate by gender
sns.barplot(x=gender_survival.index, y=gender_survival.values)
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Gender')
plt.show()

# Analyze the survival rate by passenger class
class_survival = combined_data.groupby('Pclass')['Survived'].mean()
print("Survival rate by passenger class:")
print(class_survival)

# Visualize the survival rate by passenger class
sns.barplot(x=class_survival.index, y=class_survival.values)
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Passenger Class')
plt.show()

# Analyze the survival rate by age group
age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
combined_data['AgeGroup'] = pd.cut(combined_data['Age'], bins=age_bins)
age_survival = combined_data.groupby('AgeGroup')['Survived'].mean()
print("Survival rate by age group:")
print(age_survival)

# Visualize the survival rate by age group
sns.barplot(x=age_survival.index, y=age_survival.values)
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Age Group')
plt.xticks(rotation=45)
plt.show()
