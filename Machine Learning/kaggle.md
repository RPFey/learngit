# Kaggle

## model

decision tree : use character of data to build a decision tree. Make predictions according to this tree.

## sklearn

### random forest

construct many trees and make predictions as the average of these trees

```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state = 1)
model.fit(X, y)
model.predict(..)
```

### decision tree model

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

find the lowest point of the validation curve

## Contest

### Titanic

> following notes are from notebook ---- A Data Science Framework.

数据分类

1. outcome vavriable (survive or not). This is a nominal datatype -- 1 for survival, 0 for death. The outcome variable depends whether this is a regression or classification model.
2. random unique identifiers. eg. Passenger ID, Ticket variables
3. variables for future feature engineering. eg. Name : we can get family size from sir name; SibSp : help to create from a family size.
4. nominal datatype. eg. Sex and Embarked : convert to dummy variables for computations.
5. Continuous datatype : eg. Age & Fare.

