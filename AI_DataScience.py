# Call Google Drive In Google Colab
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

# Import Pandas for Read Csv File
ds = pd.read_csv('/content/drive/My Drive/dataset/Maths.csv')
ds.head(10)

# Filtering Data
ds_result = ds[((ds['age'] > 18) & (ds['age'] < 20))]
ds_result

# Iloc Data
x = ds.iloc[0:5, 6:7]
y = ds.iloc[:, 7]
x

# Selection Data (x_train, x_test, y_train, y_test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# Result
clf = SVC()
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# Result Accuracy
print(accuracy_score(y_pred, y_test))
