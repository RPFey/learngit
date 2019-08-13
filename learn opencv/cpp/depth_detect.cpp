#include<iostream>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;
double thres = 10.0;

bool Is_sub(Mat& image);

int main(){
    Mat image = imread("0706color11.jpg");
    // GaussianBlur(depth,depth,Size(5,5),2.0,2.0);
    // filter2D(depth,out,-1,kernel);
    // Canny(depth,depth,5.0,8.0,3,1);
    // threshold(depth,depth,20.0,255,THRESH_BINARY);
    // Mat mor = getStructuringElement(MORPH_RECT,Size(1,10));
    // morphologyEx(depth,depth,MORPH_OPEN,mor);
    if(Is_sub(image)){
        
    }
    // adaptiveThreshold(depth,depth,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,5,0.0);
    // Mat mor = getStructuringElement(MORPH_RECT,Size(3,3));
    // morphologyEx(depth,depth,MORPH_OPEN,mor);
    // imwrite("morph.jpg",depth);
    return 0;
}

bool Is_sub(Mat& image){
    Mat kernel = (Mat_<float>(3,3)<<0,-1,0,-1,5,-1,0,-1,0);
   
    Mat hsv;
    cvtColor(image,hsv,COLOR_BGR2HSV);
    Mat channel[3];
    split(hsv,channel);
    Mat depth_src = channel[0].clone();
    Mat depth;
    filter2D(depth_src,depth_src,-1,kernel);
    bilateralFilter(depth_src,depth,5,0.5,5.0);
    imwrite("depth.jpg",depth);
    
    Canny(depth,depth,thres,2*thres,3,1);
    threshold(depth,depth,20.0,255,THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<std::vector<cv::Point>>::iterator it;
    std::vector<cv::Vec4i> hie;
    
    cv::findContours(depth,contours,hie,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE,cv::Point(0,0));     
    if(contours.empty()){
        cout<<"none";
        return 0;
    }
    else{
        it = contours.begin();
        while(it!=contours.end()){
            if((*it).size()<30){
                contours.erase(it,it+1);
            }
            else {it++;}
        }
        if(contours.empty()){
            cout<<"none";
            imwrite("edge_point.jpg",depth);
            return 0;
        } else{  
            for(int i=0;i<contours.size();i++){
                for(int j=0;j<contours[i].size();j++){
                    depth.at<uchar>(contours[i][j].y,contours[i][j].x)=127;
                }
                cout<<"number:   "<<contours[i].size()<<endl;
            }
            imwrite("edge_point.jpg",depth);
            return 1;
        }
    }    
}