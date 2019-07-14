#include<iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "/opencv/opencv/modules/core/include/opencv2/core.hpp"

using namespace std;
using namespace cv;

int main(){
    Mat_<Vec3b> image;
    Mat img_gry,img_canny;
    image = imread("DJI_0210.JPG");
    cout<<"column: "<<image.cols<<endl;
    cout<<"row: "<<image.rows<<endl;
    pyrDown(image,image);
    // Mat kernel_x = (Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1); // 定义卷积核
    cvtColor(image,img_gry,COLOR_BGR2GRAY);
    Canny(img_gry,img_canny,10,100,3,true);
    imshow("cann",img_canny);
    waitKey(0);
    destroyWindow("cann");
    
    return 0;
}