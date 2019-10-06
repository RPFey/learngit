开始opencv 进阶篇

# OPENCV_CONTRIB

## FEATURE EXTRACTION AND MATCHING

使用 sift 特征 

xfeature2d.sift_create()





## selective search

```python
# create Selective Search Segmentation Object using default parameters
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# set input image on which we will run segmentation
ss.setBaseImage(im)

# Switch to fast but low recall Selective Search method
ss.switchToSelectiveSearchFast()

# another mode is accurate : ss.switchToSelectiveSearchQaulity()

# run selective search segmentation on input image
rects = ss.process()
print('Total Number of Region Proposals: {}'.format(len(rects)))
```

位于 contrib 模块中



