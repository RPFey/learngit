写一些　cpp 中的用法

# vector 

## copy 
```c++
vector<int> list;
// 1. initialization
vector<int> copy_(list);
// 2. assgin (copy)
copy_.assign(list.begin(),list.end());
// 3. swap, the original one is empty
copy_.swap(list);
// 4. insert, insert the original one into the new vector
copy_.insert(copy_.end(), list.begin(), list.end());
// the first argument can change to decide where to insert
```

# Boost

## boost::array

```c++
// 与　ros 消息转化　
boost::array<double,9> intrinc {msg->K}; // K 是float64[9]
// 与　mat 转化
cv::Mat temp_intri(3,3,CV_64FC1, &intrinc[0]);
```
