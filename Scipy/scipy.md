# Scipy

* scipy.spatial.Delaunay

寻找位于凸多边形内的点

```python
def in_hull(p, hull):
    '''
    p : points (N, 3)
    hull : 多边形的顶点 (8, 3)
    '''
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Dela李白unay(hull)
    return hull.find_simplex(p)>=0

# 这里是将这个多面体划分为多个四面体，find_simplex 是找位于其中的点，并返回其所属的索引，若该点不在任何四面体中，则是 -1
```
