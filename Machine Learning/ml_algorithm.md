# Implementation of Papers

## VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Dectection

### IoU

in 3d dimensions (考虑到三维空间中立方体旋转，采用填充的方法计算 IoU)

```python
# coords of boxes are given by the corner of the box in shape (4,2)
box1_corner = ((cfg.W,cfg.H)-(box1_corner - (cfg.xrange[0], cfg.yrange[0]))/(cfg.vh, cfg.vw)).astype(np.uint32)
box2_corner = ((cfg.W,cfg.H)-(box2_corner - (cfg.xrange[0], cfg.yrange[0]))/(cfg.vh, cfg.vw)).astype(np.uint32)
# W, H are voxel number in each dimension; xrange, yrange are the truncated range of the point cloud, these two lines change to the grid coor

buf1 = np.zeros((cfg.H, cfg.W, 3))
buf2 = np.zeros((cfg.H, cfg.W, 3))
buf1 = cv2.fillConvexPoly(buf1, box1_corner, color=(1,1,1))[..., 0]
buf2 = cv2.fillConvexPoly(buf2, box2_corner, color=(1,1,1))[..., 0]

indiv = np.sum(np.abs(buf1-buf2)) # individual region
share = np.sum((buf1+buf2)==2) # share region
```

### bbox and anchor matching

有采用 cython 加快运行速度

```python
import numpy as np
cimport numpy as np # import numpy module as c

DTYPE = np.float32
ctypedef float DTYPE_t

def bbox_overlaps(
	np.ndarray[DTYPE_t, ndim=2] boxes,
	np.ndarray[DTYPE_t, ndim=2] query_boxes
):

	cdef unsigned int N = boxes.shape[0]
	cdef unsigned int K = query_boxes.shape[0]
	cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N,K), dtype=DTYPE)
	cdef DTYPE_t iw, ih, box_area
	cdef DTYPE_T ua
	cdef unsigned int k, n
	for k in range(K):
		box_area = (
			(query_boxes[k,2] - query_boxes[k,0] + 1)*
			(query_boxes[k,3] - query_boxes[k,1] + 1)
		)
	for n in range(N):
```
