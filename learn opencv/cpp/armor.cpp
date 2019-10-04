#define OPENCV_TRAITS_ENABLE_DEPRECATED
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <iostream>
#include <vector>
#include <algorithm>


bool compare_match(const cv::DMatch& a, const cv::DMatch& b){
	return a.distance<b.distance ;
}

void surf_match(const cv::Mat& temp, const cv::Mat& img, cv::Point& match){
	std::vector<cv::KeyPoint> kp1,kp2;
	cv::Mat des1,des2;
	int Hess = 100;
	cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(Hess);

	surf->detect(temp, kp1);
	surf->compute(temp, kp1, des1);
	surf->detect(img, kp2);
	surf->compute(img, kp2, des2);

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	std::vector<std::vector<cv::DMatch>> knn_matches;
	matcher->knnMatch(des1, des2, knn_matches, 2);

	const float ratio_thresh = 0.7f;
	std::vector<cv::DMatch> good_match;
	for (size_t i =0;i<knn_matches.size();i++){
		if(knn_matches[i][0].distance<ratio_thresh * knn_matches[i][1].distance){
			good_match.push_back(knn_matches[i][0]);
		}
	}
    if (good_match.size()==0){
		match.x = -1;
		match.y = -1;
	}
	else {
		std::sort(good_match.begin(), good_match.end(), compare_match);
		cv::DMatch best = good_match[0];
		int x = int(kp2[best.trainIdx].pt.x);
		int y = int(kp2[best.trainIdx].pt.y);
		match.x = x;
		match.y = y;
	}
}

void morph(const cv::Mat& img, std::vector<std::vector<cv::Point>>& result){
	cv::Mat src, src_;
	cv::bilateralFilter(img, src_, 5, 0.5, 2);
	cv::cvtColor(src_,src,cv::COLOR_BGR2GRAY);
	cv::threshold(src, src, 150, 255, cv::THRESH_BINARY);
	cv::Mat kernel;
	cv::Mat bi;
	kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1,5));
	cv::morphologyEx(src, bi, cv::MORPH_OPEN, kernel);
	kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1,3));
	cv::morphologyEx(bi, bi, cv::MORPH_CLOSE, kernel);

	std::vector<std::vector<cv::Point>> contours;
    std::vector<std::vector<cv::Point>>::iterator it;
	std::vector<cv::Vec4i> hie;
    
	
	cv::findContours(bi,contours,hie,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE,cv::Point(0,0));
	it = contours.begin();
	while(it!=contours.end()){
		if ((*it).size()>40 || (*it).size()<15){
			contours.erase(it, it+1);
			continue;
		}
		++it;
	}
	result.swap(contours);
}

void hist_calc(cv::Mat& temp, cv::Mat& img, cv::Point& track){
	cv::Mat temp_gray;
	cv::cvtColor(temp,temp_gray,cv::COLOR_BGR2GRAY);
	// cv::cvtColor(img,img_gray,cv::COLOR_BGR2GRAY);
	cv::Mat temp_hsv, src_hsv;
	cv::cvtColor(temp, temp_hsv, cv::COLOR_BGR2HSV);
	cv::cvtColor(img, src_hsv, cv::COLOR_BGR2HSV);
	int channels[] = {0,1};
	int hbins = 90, sbins = 128;
	int histSize[] = {hbins, sbins};
	float hranges[] = {0, 180};
	float sranges[] = {0, 255};
	const float* ranges[] = {hranges, sranges};
	cv::MatND hist;
	cv::calcHist(&temp_hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges);
	cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());
	cv::MatND mask;
	cv::calcBackProject(&src_hsv, 1, channels, hist, mask, ranges, 1.0, true);
	cv::threshold(mask, mask, 100, 255, cv::THRESH_BINARY);
	std::vector<int> x_pos,y_pos;
	for (int i = 0 ; i<mask.cols;i++){
		for(int j = 0; j<mask.rows;j++){
			if (mask.at<uchar>(j,i) > 0){
				x_pos.push_back(i);
				y_pos.push_back(j);
			}
		}
	}
	if(x_pos.size()>0){
		std::sort(x_pos.begin(),x_pos.end());
		std::sort(y_pos.begin(),y_pos.end());
		int x_min = x_pos[0];
		int y_min = y_pos[0];
		int x_max = x_pos[x_pos.size()-1];
		int y_max = y_pos[y_pos.size()-1];
		cv::Mat roi = img(cv::Rect(cv::Point2d(x_min, y_min),cv::Point2d(x_max, y_max))).clone();
		std::vector<std::vector<cv::Point>> contours;
		morph(roi, contours);
		cv::Point match;
		cv::Mat roi_gray;
		cv::cvtColor(roi,roi_gray,cv::COLOR_BGR2GRAY);
		surf_match(temp_gray, roi_gray, match);
		cv::rectangle(img, cv::Rect(cv::Point2d(x_min, y_min),cv::Point2d(x_max, y_max)), cv::Scalar(255,0,0), 1);
		if(match.x == -1){
			track.x = -1;
			track.y = -1;
		}
		else if (contours.size()==0){
			track.x = x_min+match.x;
			track.y = y_min+match.y;
		}
		else{
			// 形状特征对匹配特征加一个约束
			std::vector<int> p_x;
			std::vector<int> p_y;
			for(int i=0;i<contours.size();i++){
				for(int j=0; j<contours[i].size();j++){
					p_x.push_back(contours[i][j].x);
					p_y.push_back(contours[i][j].y);
				}
			}
			std::sort(p_x.begin(),p_x.end());
			std::sort(p_y.begin(),p_y.end());
			int _x_min = p_x[0];
			int _x_max = p_x[p_x.size()-1];
			int _y_min = p_y[0];
			int _y_max = p_y[p_y.size()-1];
			if(match.x>_x_min&&match.x<_x_max&&match.y>_x_min&&match.y<_x_max){
				track.x = x_min+match.x;
				track.y = y_min+match.y;
			} else{
				track.x = (_x_max+_x_min)/2;
				track.y = (_y_min+_y_max)/2;
			}
		}
	}
}

void video_test(){
	cv::VideoCapture video("5armor1.mp4");
	cv::Mat template_bar = cv::imread("80_extract.jpg");

	cv::Mat frame,frame_gray;
	video>>frame;
	if(frame.empty()){
		return ;
	}
	cv::cvtColor(frame,frame_gray, cv::COLOR_BGR2GRAY);
	cv::Mat old_gray(frame_gray);
	std::vector<cv::Point2f> old_point,new_point;
	float error=0.0;
	cv::namedWindow("track",cv::WINDOW_AUTOSIZE);
	while(true){
		
		video>>frame;
		if(frame.empty()){
			break;
		}
		cv::cvtColor(frame,frame_gray, cv::COLOR_BGR2GRAY);

		if(old_point.size()==0){
			cv::Point track;
			hist_calc(template_bar,frame, track);
			
			if(track.x == -1){
				old_point.clear();
				continue;
			}
			cv::Point2f new_point_;
			new_point_.x = (float)track.x;
			new_point_.y = (float)track.y;
			new_point.push_back(new_point_);
			old_point = new_point;
			std::cout<<new_point;
			error = 0.0;
		} else {
			std::vector<uchar> status;
			std::vector<uchar> err;
			cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT)+(cv::TermCriteria::EPS), 10, 0.03);
			cv::calcOpticalFlowPyrLK(old_gray, frame_gray, old_point, new_point, status, err, cv::Size(15,15), 3, criteria);
			
			std::vector<cv::Point2f> good_new;
			old_point.clear();
			for(uint i =0 ; i<old_point.size(); i++){
				if(status[i]==1){
					good_new.push_back(new_point[i]);
					cv::circle(frame, new_point[i],10, cv::Scalar(0,0,255),-1);
				}
			}
			if (good_new.size()>0){
				old_point.push_back(good_new[0]);
			}
		}
		old_gray = frame_gray.clone();
		cv::imshow("track", frame);
		cv::waitKey(60);
	}
}
int main(){
	video_test();
	return 0;
}
