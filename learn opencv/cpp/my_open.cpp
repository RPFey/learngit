#include<iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "/opencv/opencv/modules/core/include/opencv2/core.hpp"

using namespace std;
using namespace cv;

int main(){
    Mat image=imread("DJI_0210.JPG");
    cvtColor(image,image,COLOR_BGR2GRAY);

    if(image.empty()){
        cout<<"can't open";
        return -1;
    }
    Mat border;
    // int y = image.cols;
    // int x = image.rows;
    // int padding = abs(x-y)/2;
    // if (x<y){
    //     copyMakeBorder(image,border,padding,padding,0,0,BORDER_WRAP);
    // }else{
    //     copyMakeBorder(image,border,0,0,padding,padding,BORDER_WRAP);
    // }
    namedWindow("border",WINDOW_AUTOSIZE);
    imshow("border",image);
    waitKey(0);
    return 0;
}