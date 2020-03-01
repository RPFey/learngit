#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
using namespace cv;
using namespace std;

int main(){
    VideoCapture vi("5armor1.mp4");
    cout<<vi.isOpened()<<endl;
    namedWindow("frame",WINDOW_AUTOSIZE);
    Mat frame;
    int i = 1;
    while(vi.isOpened()) {
        vi.read(frame);
        imshow("frame", frame);
        char s = waitKey(50);
        if (s == 's'){
            char path[100];
            sprintf(path,".//capture//%d.jpg",i);
            imwrite(path,frame);
            i++;
        }
    }
    return 0;
}

