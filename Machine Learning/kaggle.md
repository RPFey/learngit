# model

decision tree : use character of data to build a decision tree. Make predictions according to this tree.

# pandas

```python
 # read files
 data = pd.read_csv("path_to_csv")
 data.describe()
```

the describe() method will show count(number of non-missing value in the set), mean, std, min, 25%, 50%, 75%, max value

```python
# drop out not available data
data.dropna(axis = 0)  # drop out row if it exists missing data.
data.columns # all the column tag we can access

# index
y = data.Price   # the Price column 

features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude'] 
# string is necessary
X = data[features]
# check first n data
X.head(n)
```

# sklearn

## random forest

construct many trees and make predictions as the average of these trees

```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state = 1)
model.fit(X, y)
model.predict(..)
```

## decision tree model

```python
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state = 1)
model.fit(X, y)
prediction = model.predict(X.head()) # X is also right
```

初始化时可选参数 ：

max_leaf_node : 控制模型大小

## validation

Mean Absolute Error (MAE)

```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, model.predict(X))
```

validation set

```python
from sklearn.model_selection import train_test_split
train_X, val_X, train_Y, val_Y = train_test_split(X, y, random_state = 1)

# leet code 欣赏
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get) # 传入字典的 get 方法作为比较的准则
```

![avatar](./img/overfit_vs_underfit.png)

find the lowest point of the validation curve