#include<iostream>
#include "opencv2/opencv.hpp"
#include<vector>
using namespace cv;
using namespace std;
int main(){
    Mat kernel = (Mat_<uchar>(3,3,CV_8UC3)<<0,0,255,1,2,45,2,4,89,\
                                          2,43,123,2,3,11,2,4,5);
    cout<<kernel<<endl;
    return 0;
}