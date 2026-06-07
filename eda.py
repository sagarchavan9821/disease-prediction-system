from database import df
import matplotlib.pyplot as plt
import seaborn as sns

print(df.shape)
print(df.info())
print(df.duplicated().sum())
print(df.isnull().sum())


#sns.countplot(data=df, x="family_history", hue="kidney_disease")


print(plt.show())
# there is no null values and duplicate  values very colume is importanat for  dicease predictaion 
