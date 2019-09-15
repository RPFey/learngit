#include<iostream>
#include "opencv2/opencv.hpp"
#include<vector>
#include<algorithm>
#include<cmath>
void Harris_demo(cv::Mat& src,std::vector<cv::Point2d>& corner,int H_thres=30);
int thres = 1;
cv::RNG rng(1234);
const float pi = 3.1415926;
const double harris_thres = 10.0;
typedef struct Shape{
    int b ;
    float area ;
}  shape; 

bool compare(shape& a,shape& b)
{
    if(a.b<b.b){
        return true;
    } else if (a.b==b.b&&a.area<b.area){
        return true;
    } else {
        return false;
    }
}

float point_dis(const cv::Point& a ,const cv::Point& b){
    float x1 = (float)a.x;
    float y1 = (float)a.y;
    float x2 = (float)b.x;
    float y2 = (float)b.y;
    float dis = std::sqrt(std::pow(x1-x2,2)+std::pow(y1-y2,2));
    return dis;
}

void filterpoint(std::vector<cv::Point>& con_shape,double thres){
    std::vector<cv::Point>::iterator a,b;
    for(a=con_shape.begin();a!=con_shape.end();a++){
        b = a+1;
        while(b!=con_shape.end()){
            if(point_dis(*a,*b)<thres){
                con_shape.erase(b,b+1);
            }else{
                b++;
            }
        }
    }
}

double get_tri_area(cv::Point&a , cv::Point& b, cv::Point& c){
    float al=point_dis(a,b),bl=point_dis(b,c),cl=point_dis(a,c);
    float p = (al+bl+cl)/2;
    float area = std::sqrt(p*(p-al)*(p-bl)*(p-cl));
    return area;
}

void print_info(std::vector<shape> out){
    for(int i =0;i<out.size();i++){
        std::cout<<"边数为：  "<<out[i].b<<"     面积为："<<out[i].area<<std::endl;
    }
}
// 绘制函数
void draw_rec(cv::Mat& src,cv::Point2d left_top,cv::Point2d right_down){
    if(src.channels()==3){
        cv::rectangle(src,cv::Rect(left_top,right_down),cv::Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)),-1);
    } else if (src.channels()==1){
        cv::rectangle(src,cv::Rect(left_top,right_down),cv::Scalar(0),5);
    } else {
        std::cout<<"channel wrong !";
        return ;
    }
}
void draw_cir(cv::Mat& src,cv::Point2d center,double r){
    if(src.channels()==3){
        cv::circle(src,center,r,cv::Scalar(rng.uniform(1,255),rng.uniform(1,255),rng.uniform(1,255)),-1);    
    }
    else if(src.channels()==1){
        cv::circle(src,center,r,cv::Scalar(0),5);
    }else {
        std::cout<<"channel wrong!";
        return ;
    }
}
void draw_tri(cv::Mat& src, cv::Point2d a, cv::Point2d b, cv::Point2d c){
    std::vector<cv::Point> pts{a,b,c};
    if(src.channels()==3){
        cv::Scalar color(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
        cv::fillConvexPoly(src,pts,color);
    } 
    else if (src.channels()==1){
        cv::line(src,a,b,cv::Scalar(0),5);
        cv::line(src,b,c,cv::Scalar(0),5);
        cv::line(src,c,a,cv::Scalar(0),5);
    }
    else {
        std::cout<<"channel wrong!";
        return ;
    }
}

void generate(cv::Mat& src){
    int row = src.rows;
    int col = src.cols;
    int block_x = col/3;
    int block_y = row/2;
    draw_tri(src,cv::Point2d(rng.uniform(0,block_x),rng.uniform(0,block_y)),cv::Point2d(rng.uniform(0,block_x),rng.uniform(0,block_y)),cv::Point2d(rng.uniform(0,block_x),rng.uniform(0,block_y)));
    draw_tri(src,cv::Point2d(rng.uniform(block_x,2*block_x),rng.uniform(0,block_y)),cv::Point2d(rng.uniform(block_x,2*block_x),rng.uniform(0,block_y)),cv::Point2d(rng.uniform(block_x,2*block_x),rng.uniform(0,block_y)));
    draw_tri(src,cv::Point2d(rng.uniform(2*block_x,3*block_x),rng.uniform(0,block_y)),cv::Point2d(rng.uniform(2*block_x,3*block_x),rng.uniform(0,block_y)),cv::Point2d(rng.uniform(2*block_x,3*block_x),rng.uniform(0,block_y)));
    int cen_x = rng.uniform(150,block_x-150),cen_y = rng.uniform(block_y+150,2*block_y-150);
    int r = rng.uniform(150,std::min(std::min(cen_x,block_x-cen_x),std::min(cen_y-block_y,2*block_y-cen_y)));
    draw_cir(src,cv::Point2d(cen_x,cen_y),(double)r);
    cen_x = rng.uniform(block_x+150,2*block_x-150);cen_y = rng.uniform(block_y+150,2*block_y-150);
    r = rng.uniform(150,std::min(std::min(cen_x-block_x,2*block_x-cen_x),std::min(cen_y-block_y,2*block_y-cen_y)));
    draw_cir(src,cv::Point2d(cen_x,cen_y),(double)r);
    draw_rec(src,cv::Point2d(rng.uniform(2*block_x,3*block_x),rng.uniform(block_y,2*block_y)),cv::Point2d(rng.uniform(2*block_x,3*block_x),rng.uniform(block_y,2*block_y)));
}
bool no_corner_inside(const std::vector<cv::Point2d>& corner,int range[]){
    for(int i=0;i<corner.size();i++){
        if(corner[i].x>=range[0]-harris_thres&&corner[i].x<=range[2]+harris_thres&&corner[i].y>=range[1]-harris_thres&&corner[i].y<=range[3]+harris_thres){
            return false;
        }
    }
    return true;
}
// 寻找凸几何形状
void Contour(cv::Mat& src){
    cv::Mat init = src.clone();
    if (src.channels()==3){
        cv::cvtColor(init,src,cv::COLOR_BGR2GRAY);
        std::cout<<"彩色！"<<std::endl;
    }else{
        std::cout<<"灰白！"<<std::endl;
    }
    threshold(src,src,254.9,255,CV_THRESH_BINARY_INV);
    std::vector<shape> out_info; 
    cv::Mat canny_out;
    cv::Canny(src,canny_out,thres,2*thres,3,1);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<std::vector<cv::Point>>::iterator it;
    std::vector<cv::Vec4i> hie;
    std::vector<cv::Point2d> corner;
    std::cout<<"harris";
    Harris_demo(src,corner);
    cv::findContours(canny_out,contours,hie,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE,cv::Point(0,0));
    if(contours.empty()){
        std::cout<<"can't find";
        return;
    }else{
        std::cout<<"here 2";
    }
    // 将顶点个数不足的图形舍去
    it = contours.begin();
    while(it!=contours.end()){
        if((*it).size()<3){
            contours.erase(it,it+1);
        }
        it++;
    }
    std::vector<std::vector<cv::Point>> contour_poly(contours.size());
    for(int i = 0; i < contours.size(); i++){
        cv::approxPolyDP(cv::Mat(contours[i]),contour_poly[i], 30, 1);
        filterpoint(contour_poly[i],10.0);
    }
    if(contour_poly.empty()){
        std::cout<<"can't find";
        return;
    } 
    it = contour_poly.begin();
    while(it!=contour_poly.end()){
        if((*it).size()<3){
            contour_poly.erase(it,it+1);
        }
        it++;
    }
    
    for(int i = 0;i<contour_poly.size();i++){
        if(contour_poly[i].size()>4){
circle:     cv::Point2f Center;
            float r;
            cv::minEnclosingCircle(contour_poly[i],Center,r);
            float area = pi*r*r;
            shape a;
            a.b = 1;
            a.area = area;
            out_info.push_back(a);
            cv::circle(init,Center,r,cv::Scalar(127),2);
        }
        else {
            int min_x=src.cols,max_x=0,min_y=src.rows,max_y=0;
            for(int j=0;j<contour_poly[i].size();j++){
                int x = contour_poly[i][j].x;
                int y = contour_poly[i][j].y;
                min_x = min_x>x?x:min_x;
                max_x = max_x>x?max_x:x;
                min_y = min_y>y?y:min_y;
                max_y = max_y>y?max_y:y;
            }
            int range[] {min_x,min_y,max_x,max_y};
            if (no_corner_inside(corner,range)){
                goto circle ;
            }          
            if(contour_poly[i].size()==3){
                float area=get_tri_area(contour_poly[i][0],contour_poly[i][1],contour_poly[i][2]);        
                shape a;
                a.b=3;
                a.area = area;
                out_info.push_back(a);    
            } else{
                float area = cv::contourArea(contour_poly[i]);
                shape a;
                a.b=4;
                a.area = area;
                out_info.push_back(a);
            }
            cv::rectangle(init,cv::Rect(cv::Point(min_x,min_y),cv::Point(max_x,max_y)),cv::Scalar(155),2); 
        }   
    }
    std::sort(out_info.begin(),out_info.end(),::compare);
    print_info(out_info);   
    cv::namedWindow("contoures",CV_WINDOW_AUTOSIZE);
    cv::imshow("contoures.jpg",init);
}

void Harris_demo(cv::Mat& src,std::vector<cv::Point2d>& corner,int H_thres){
    corner.clear();
    cv::Mat dst = cv::Mat::zeros(src.size(),CV_32FC1);
    cv::Mat grey;
    cv::Mat normImage;
    if(src.channels()!=1){
    cv::cvtColor(src,grey,cv::COLOR_BGR2GRAY);
    } else{
        grey = src.clone();
    }
    cv::threshold(grey,grey,254,255,cv::THRESH_BINARY_INV);
    cv::cornerHarris(grey,dst,2,3,0.04,cv::BORDER_DEFAULT);
    cv::normalize(dst,normImage,0,255,cv::NORM_MINMAX,CV_32FC1);
    //将归一化后的图线性变换成8位无符号整形
	
	//将检测到的，且符合阀值条件的角点绘制出来
	for (int i = 0; i < normImage.rows; i++)
	{
		for (int j = 0; j < normImage.cols; j++)
		{
			if ((int)normImage.at<float>(i, j) > H_thres + 80)
			{
				cv::circle(src, cv::Point(j, i), 20, cv::Scalar(0, 10, 255), 2, 8, 0);
                corner.push_back(cv::Point2d(j,i));
			}
		}
	}
}
int main(){
    // 彩色填充图像
    cv::Mat color_can(cv::Size(1440,1080),CV_8UC3,cv::Scalar(255,255,255));
    //generate(color_can);
    draw_cir(color_can,cv::Point(500,500),30);
    draw_rec(color_can,cv::Point(100,100),cv::Point(400,500));
    //Harris_demo(color_can,30);
    cv::namedWindow("col_drawing",CV_WINDOW_AUTOSIZE);
    cv::imshow("col_drawing",color_can);
    Contour(color_can);
    cv::waitKey(0);
    return 0;
}




/* ------------------   之前随便写的一些 不用管 --------------------- */
// Mat detect_zhi(Mat& src){
//     Mat zhi(src.size(),src.type());
//     bitwise_not(src,src);
//     GaussianBlur(src,zhi,Size(5,5),2);
//     cout<<int(zhi.at<uchar>(150,599))<<endl;
//     cout<<int(zhi.at<uchar>(500,700))<<endl;
//     // Mat kernel = (Mat_<float>(3,3)<<0,0,0,0,0.3,0.3,0,0.3,0);
//     // filter2D(src,zhi,-1,kernel);
//     return zhi.clone();
// }
// void add_result(Mat& zhi){
//     for(int row=0;row<zhi.rows;row++){
//         for(int col=0;col<zhi.cols;col++){
//             int i = zhi.at<uchar>(row,col);
//             if (i>100){
//                 cout<<"value:"<<i<<"  "<<"row :"<<row<<"   "<<"col: "<<col<<endl;
//                 circle(zhi,Point2d(col,row),5,Scalar(255),-1);
//             }
//         }
//     }
// }