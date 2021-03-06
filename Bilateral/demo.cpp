#include <iostream>
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include<sstream>
#include "Bilateral.h"

using namespace cv;

VideoCapture video;
VideoWriter videowriter;
std::vector<Mat> imgSrcArr, maskArr;
Mat imgShow; //原图与展示图
std::vector<Point> forePts, backPts; //vector相当于数组，分别存储前景点和背景点
bool IsLeftDown = false, IsRightDown = false;  //定义了两个假
Point currentPoint, nextPoint;

static void OnMouse(int even, int x, int y, int flags, void* param) //调用鼠标，总地来说就是收集前景节点和背景节点保存在g_fore/backPts中，并实时显示
{
	int xMax = video.get(CV_CAP_PROP_FRAME_WIDTH)-1;
	int yMax = video.get(CV_CAP_PROP_FRAME_HEIGHT)-1;
	x = x >= 0 ? x : 0;
	x = x <= xMax ? x : xMax;
	y = y >= 0 ? y : 0;
	y = y <= yMax ? y : yMax;

	if (even == CV_EVENT_LBUTTONDOWN) //鼠标左键点击
	{
		currentPoint = Point(x, y);
		forePts.push_back(currentPoint);
		IsLeftDown = true;
		return;
	}
	if (IsLeftDown&&even == CV_EVENT_MOUSEMOVE)//鼠标左键按着没松还在动
	{
		nextPoint = Point(x, y);
		forePts.push_back(nextPoint);
		line(imgShow, currentPoint, nextPoint, Scalar(255, 0, 0), 2); // 255.0.0是蓝色  10是线条宽度，类似于把鼠标点为中心10单位半径内的点归为point
		currentPoint = nextPoint;
		imshow("原图像", imgShow);
		return;
	}
	if (IsLeftDown&&even == CV_EVENT_LBUTTONUP) //松了
	{
		IsLeftDown = false;
		return;
	}

	if (even == CV_EVENT_RBUTTONDOWN) //按下右键
	{
		currentPoint = Point(x, y);
		backPts.push_back(currentPoint);
		IsRightDown = true;
		return;
	}
	if (IsRightDown&&even == CV_EVENT_MOUSEMOVE)//划线
	{
		nextPoint = Point(x, y);
		backPts.push_back(nextPoint);
		line(imgShow, currentPoint, nextPoint, Scalar(0, 0, 255), 2);
		currentPoint = nextPoint;
		imshow("原图像", imgShow);
		return;
	}
	if (IsRightDown&&even == CV_EVENT_RBUTTONUP)//松了
	{
		IsRightDown = false;
		return;
	}
}


int main() {
	video.open("image/parkour.avi");
	videowriter.open("image/output.avi", CV_FOURCC('D', 'I', 'V', 'X'), 5, Size(video.get(CV_CAP_PROP_FRAME_WIDTH), video.get(CV_CAP_PROP_FRAME_HEIGHT)));
	//CAP_PROP_FRAME_COUNT
	for (int times = 0; times < 1; times++)
	{
		for (int i = 0;i < 10;i++) {
			Mat imgSrc;
			video >> imgSrc;
			imgSrcArr.push_back(imgSrc);
		}
		forePts.clear();
		backPts.clear();
		imgSrcArr[5].copyTo(imgShow);//备份一份
		namedWindow("原图像");
		setMouseCallback("原图像", OnMouse);//鼠标调用函数

		while (1)
		{
			imshow("原图像", imgShow);
			int t = waitKey();
			if (t == 27) break; //27就是esc,随时生效

			char c = (char)t;
			if (c == 's')   //键盘输入S实现分割
			{
				printf("\n第%d次分割\n", times+1);
				double _time = static_cast<double>(getTickCount());
				Bilateral bilateral(imgSrcArr);
				bilateral.InitGmms(forePts, backPts, 5);
				bilateral.run(maskArr);
				_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
				printf("总用时为%f\n", _time);//显示时间
				
				for (int t = 0; t < imgSrcArr.size(); t++)
				{
					Mat mask = maskArr[t];
					Mat maskBlur, lastImg;
					medianBlur(mask, maskBlur, 5);
					imgSrcArr[t].copyTo(lastImg, maskBlur);
					videowriter << lastImg;
				}

				//清除容器，释放内存
				for (int i = imgSrcArr.size() - 1;i >= 0;i--) {
					imgSrcArr[i].release();
				}
				imgSrcArr.clear();
				for (int i = maskArr.size() - 1;i >= 0;i--) {
					maskArr[i].release();
				}
				maskArr.clear();
			}
		}
	}

	videowriter.release();
	video.release();
	imgSrcArr.clear();
	imgShow.release();
	destroyAllWindows();
	return 0;
}