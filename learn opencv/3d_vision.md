# triangulatePoints()

三角法测距

```c++
cv::Mat pts_4d;
cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);
for (int i = 0; i < pts_4d.cols; i++) {
    cv::Mat x = pts_4d.col(i);
    x /= x.at<float>(3, 0); // 归一化
    cv::Point3d p(
        x.at<float>(0, 0),
        x.at<float>(1, 0),
        x.at<float>(2, 0)
    );
    points.push_back(p);
}
```

注意处理方法，pts_4d 是一个四维的其次坐标点，采用Mat 类型来接受。

