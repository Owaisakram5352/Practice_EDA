# Insurance Claim Prediction

This project aims to predict insurance claim likelihood using a given dataset. The repository includes the code and analysis performed to achieve this goal.

## Libraries Used

- pandas: Data manipulation and analysis library
- numpy: Numerical computing library
- matplotlib: Data visualization library
- seaborn: Statistical data visualization library
- scikit-learn: Machine learning library for predictive modeling

## Dataset

The dataset used in this project is named "train_qWM28Yl.csv". It contains information related to insurance policies, including features such as policy ID, claim status, and various attributes. The dataset is stored in a CSV file format.

### Basic Information

After loading the dataset, basic information was obtained using the following commands:

df = pd.read_csv("/content/train_qWM28Yl.csv")
df.info()
df.describe()

### Data Quality Checks
df.isnull().sum()
df.duplicated().sum()
### Exploratory Data Analysis (EDA)
sns.countplot(x=df.is_claim, data=df)
num_col = [col for col in df.columns if df[col].dtype != "object"]
for col in num_col:
    plt.figure()
    sns.boxplot(x="is_claim", y=df[col], data=df)
    plt.xlabel("is_claim")
    plt.ylabel(col)
    plt.title('Box Plot for ' + col)
 ### Feature Engineering
Label encoding was performed on categorical features using scikit-learn's LabelEncoder() to convert them into numerical representations.
