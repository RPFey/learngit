#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

void OnMouse(int event,int x,int y,int flags,void *ustc);
Mat* p;
void edge_detect(Mat& );
int main(){
    Mat car = imread("80.jpg");
    namedWindow("raw",WINDOW_AUTOSIZE);
    imshow("raw",car);
    // edge_detect(car);
    // Mat hsv;
    // cvtColor(car,hsv,COLOR_BGR2HSV);
    p = &car;
    namedWindow("data",WINDOW_AUTOSIZE);
    imshow("data",car);
    setMouseCallback("data",OnMouse);
    waitKey();
    system("pause");
    // cv::waitKey(0);
    return 0;
}

void OnMouse(int event,int x,int y, int flags, void* ustc){
    Mat image = *p;
    if(event == CV_EVENT_LBUTTONDBLCLK){
        cout<<"x:   "<<x<<"   y: "<<y<<endl;
    }
}

void edge_detect(Mat& car){
    Mat grey,grey1;  
    cvtColor(car,grey1,COLOR_BGR2GRAY);
    threshold(grey1,grey1,40,255,THRESH_BINARY);
    Mat open_ker;
    open_ker = getStructuringElement(MORPH_RECT,Size(13,13));
    morphologyEx(grey1,grey1,MORPH_TOPHAT,open_ker,Point(-1,-1),1);
    open_ker = getStructuringElement(MORPH_RECT,Size(7,7));
    morphologyEx(grey1,grey1,MORPH_OPEN,open_ker);
    open_ker = getStructuringElement(MORPH_RECT,Size(5,5));
    morphologyEx(grey1,grey1,MORPH_CLOSE,open_ker);
    namedWindow("MORPH",WINDOW_AUTOSIZE);
    imshow("MORPH",grey1);
    bilateralFilter(grey1,grey,7,1,5);
    Canny(grey,grey,20,40,3,true);
    namedWindow("edge",WINDOW_AUTOSIZE);
    imshow("edge",grey);
    threshold(grey,grey,127,255,THRESH_BINARY);
    std::vector<std::vector<cv::Point>> contours;
    // std::vector<std::vector<cv::Point>>::iterator it;
    std::vector<cv::Vec4i> hie;
    int index=-1;
    std::vector<cv::Point2d> corner;
    cv::findContours(grey,contours,hie,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE,cv::Point(0,0));
    for(int i=0;i<contours.size();i++){      
        if(contours[i].size()>200){
            index = i;
        }
    }
    if (index == -1){
        return ;
    }
    int x_l = 10000,x_h = -1,y_l=10000, y_h = -1;
    for (int i=0;i<contours[index].size();i++){
        x_h = max(x_h,contours[index][i].x);
        x_l = min(x_l,contours[index][i].x);
        y_l = min(y_l,contours[index][i].y);
        y_h = max(y_h,contours[index][i].y);
    }
    namedWindow("contour_point",WINDOW_AUTOSIZE);
    imshow("contour_point",grey);
    Mat sub = grey(Rect(Point(x_l-5,y_l-5),Point(x_h+5,y_h+5)));
    namedWindow("subtract",WINDOW_AUTOSIZE);
    imshow("subtract",sub);
}