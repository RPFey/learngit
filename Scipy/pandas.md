# pandas

## overview of data

> It's a good idea to use tools below to take a glimpse of data before starting.

```python
 # read files
 data = pd.read_csv("path_to_csv")
 data.describe()
```

the describe() method will show count(number of non-missing value in the set), mean, std, min, 25%, 50%, 75%, max value

```python
data.info()
```

info() method will show the type and number of collected data points.

```python
data_raw.head()
data_raw.tail()
data_raw.sample(10)
```

These methods will show part of the data points.

## Data perparation

* detect missing values

pandas provides .isnull() method to find Nan elements.

```python
print(data.isnull().sum())
```

> isnull() will return the bool type array-like object with the same size

* Clean ? or complete ?

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

.fillna() method can complete missing values

```python
dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
```

> value could be scalar like .median(), but also like dicts -- fill different indexs with different values or DataFrame

## Basic Operation

* index 

感觉和 numpy.ndarray 差不多。二维数组索引。只不过除了用数字外，可以用字符串索引（对应行列的 index）

## Documentation

### I/O operation

* read_csv()

其中有一个参数`seq`作为分割符，将输入`.csv`文件拆分为列表。