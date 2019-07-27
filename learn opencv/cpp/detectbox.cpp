#include<iostream>
#include "opencv2/opencv.hpp"
#include<vector>
using namespace cv;
using namespace std;
int thres = 100;
void draw_rec(Mat& src,Point2d left_top,Point2d right_down){
    if(src.channels()!=1){
        cout<<"channel wrong!";
        return;
    }
    rectangle(src,Rect(left_top,right_down),Scalar(0),5);
}
void draw_cir(Mat& src,Point2d center,double r){
    if(src.channels()!=1){
        cout<<"channel wrong!";
        return ;
    }
    circle(src,center,r,Scalar(0),5);
}
void draw_tri(Mat& src, Point2d a, Point2d b, Point2d c){
    if(src.channels()!=1){
        cout<<"channel wrong!";
        return ;
    }
    line(src,a,b,Scalar(0),5);
    line(src,b,c,Scalar(0),5);
    line(src,c,a,Scalar(0),5);
}

void Contour(Mat& src){
    Mat canny_out;
    vector<vector<Point>> contours;
    vector<vector<Point>>::iterator it;
    vector<Vec4i> hie;
    Canny(src,canny_out,thres,thres*2,3);
    findContours(canny_out,contours,hie,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE,Point(0,0));
    Mat drawing = Mat(canny_out.size(),CV_8UC1,Scalar(255));
    // 将个数不足的边舍去
    for(it = contours.begin();it!=contours.end();it++){
        if((*it).size()<3){
            contours.erase(it,it+1);
        }
    }
    vector<vector<Point>> contour_poly(contours.size());
    for(int i = 0; i < contours.size(); i++){
        approxPolyDP(Mat(contours[i]),contour_poly[i], 30, 1);
        int min_x=src.cols,max_x=0,min_y=src.rows,max_y=0;
        for(int j=0;j<contour_poly[i].size();j++){
            int x = contour_poly[i][j].x;
            int y = contour_poly[i][j].y;
            min_x = min_x>x?x:min_x;
            max_x = max_x>x?max_x:x;
            min_y = min_y>y?y:min_y;
            max_y = max_y>y?max_y:y;
        }
        cout<<"边数： "<<contour_poly[i].size()<<"  顶点： "<<Point(min_x,min_y)<<" ; "<<Point(max_x,max_y)<<endl;
        rectangle(src,Rect(Point(min_x,min_y),Point(max_x,max_y)),Scalar(155),2);    
    }    
    namedWindow("contoures",CV_WINDOW_AUTOSIZE);
    imshow("contoures",src);
}

Mat detect_zhi(Mat& src){
    Mat zhi(src.size(),src.type());
    bitwise_not(src,src);
    GaussianBlur(src,zhi,Size(5,5),2);
    cout<<int(zhi.at<uchar>(150,599))<<endl;
    cout<<int(zhi.at<uchar>(500,700))<<endl;
    // Mat kernel = (Mat_<float>(3,3)<<0,0,0,0,0.3,0.3,0,0.3,0);
    // filter2D(src,zhi,-1,kernel);
    return zhi.clone();
}
void add_result(Mat& zhi){
    for(int row=0;row<zhi.rows;row++){
        for(int col=0;col<zhi.cols;col++){
            int i = zhi.at<uchar>(row,col);
            if (i>100){
                cout<<"value:"<<i<<"  "<<"row :"<<row<<"   "<<"col: "<<col<<endl;
                circle(zhi,Point2d(col,row),5,Scalar(255),-1);
            }
        }
    }
}
int main(){
    Mat canvas(Size(1440,1080),CV_8UC1,Scalar(255));
    draw_rec(canvas,Point2d(500,500),Point2d(700,700));
    draw_cir(canvas,Point2d(100,100),200.0);
    draw_tri(canvas,Point2d(150,150),Point2d(500,150),Point2d(150,600));
    namedWindow("drawing",CV_WINDOW_AUTOSIZE);
    imshow("drawing",canvas);

    // Mat zhi(canvas.size(),canvas.type(),Scalar(255));
    // Mat zhi = detect_zhi(canvas);
    // add_result(zhi);
    // Mat kernel = (Mat_<float>(3,3)<<0,0,0,0,0.3,0.3,0,0.3,0);
    // filter2D(canvas,zhi,-1,kernel);
    // namedWindow("zhi",CV_WINDOW_AUTOSIZE);
    // imshow("zhi",zhi);
    Contour(canvas);
    waitKey(0);
    return 0;
}