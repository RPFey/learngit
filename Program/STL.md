# STL

## vector

### copy

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

`.data()` returns the pointer to the first element.