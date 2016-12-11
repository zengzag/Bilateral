#include <iostream>
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include "slic.h"
#include "Bilateral.h"
#include "BilateralSimple.h"

using namespace std;
using namespace cv;

VideoCapture video;
VideoWriter videowriter;
std::vector<Mat> imgSrcArr, maskArr, keyMaskArr;

const Scalar RED = Scalar(0, 0, 255);
const Scalar PINK = Scalar(230, 130, 255);
const Scalar BLUE = Scalar(255, 0, 0);
const Scalar LIGHTBLUE = Scalar(255, 255, 160);
const Scalar GREEN = Scalar(0, 255, 0);

const int BGD_KEY = EVENT_FLAG_CTRLKEY;
const int FGD_KEY = EVENT_FLAG_SHIFTKEY;

static void help()
{
	cout << "\nSelect a rectangular area around the object you want to segment\n" <<
		"\nHot keys: \n"
		"\tESC - quit the program\n"
		"\ts - start\n"
		"\n"
		"\tleft mouse button - set rectangle\n"
		"\n"
		"\tCTRL+left mouse button - set GC_BGD pixels\n"
		"\tSHIFT+left mouse button - set GC_FGD pixels\n"
		"\n"
		<< endl;
}



static void getBinMask(const Mat& comMask, Mat& binMask)
{
	if (comMask.empty() || comMask.type() != CV_8UC1)
		CV_Error(Error::StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
	if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols)
		binMask.create(comMask.size(), CV_8UC1);
	binMask = comMask & 2;
}

class GCApplication
{
public:
	const string* winName;//窗口名
	const Mat* image; //输入图
	Mat mask;
	Mat res;
	uchar rectState, lblsState, prLblsState;
	bool isInitialized;
	Rect rect;

public:
	enum { NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
	static const int radius = 2;
	static const int thickness = -1;

	void reset();
	void setImageAndWinName(const Mat& _image, const string& _winName);
	void showImage() const;
	void mouseClick(int event, int x, int y, int flags, void* param);
	void notSetRect();
private:
	void setRectInMask();
	void setLblsInMask(int flags, Point p, bool isPr);
	void reLblsInMask(Point pCurrent, Point pCenter, bool isFGD);
};

void GCApplication::reset()
{
	if (!mask.empty())
		mask.setTo(Scalar::all(GC_PR_FGD));
	if (!res.empty())
		image->copyTo(res);
	isInitialized = false;
	rectState = NOT_SET;
	lblsState = NOT_SET;
	prLblsState = NOT_SET;
}

void GCApplication::notSetRect() {
	rectState = SET;
}

void GCApplication::setImageAndWinName(const Mat& _image, const string& _winName)
{
	if (_image.empty() || _winName.empty())
		return;
	image = &_image;
	winName = &_winName;
	mask.create(image->size(), CV_8UC1);
	image->copyTo(res);
	reset();
}

void GCApplication::showImage() const
{

	if (rectState == IN_PROCESS) {
		Mat temp;
		res.copyTo(temp);
		rectangle(temp, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), BLUE, -1);
		imshow(*winName, temp);
	}
	else if (rectState == SET) {
		rectangle(res, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), BLUE, -1);
		imshow(*winName, res);
	}
	else {
		imshow(*winName, res);
	}
}

void GCApplication::setRectInMask()
{
	CV_Assert(!mask.empty());
	rect.x = max(0, rect.x);
	rect.y = max(0, rect.y);
	rect.width = min(rect.width, image->cols - rect.x);
	rect.height = min(rect.height, image->rows - rect.y);
	(mask(rect)).setTo(Scalar(GC_BGD));
}

void GCApplication::setLblsInMask(int flags, Point p, bool isPr)
{
	Scalar bpxls, fpxls;
	uchar bvalue, fvalue;
	if (!isPr)
	{
		bpxls = BLUE;
		fpxls = RED;
		bvalue = GC_BGD;
		fvalue = GC_FGD;
	}
	else
	{
		bpxls = LIGHTBLUE;
		fpxls = PINK;
		bvalue = GC_BGD;
		fvalue = GC_FGD;
	}
	if (flags & BGD_KEY)
	{
		circle(res, p, radius, LIGHTBLUE, thickness);;
		circle(mask, p, radius, bvalue, thickness);
	}
	if (flags & FGD_KEY)
	{
		circle(res, p, radius, PINK, thickness);;
		circle(mask, p, radius, fvalue, thickness);
	}
}

void GCApplication::reLblsInMask(Point pCurrent, Point pCenter, bool isFGD)
{
	uchar value = isFGD ? GC_FGD : GC_BGD;
	Scalar pxls = isFGD ? RED : BLUE;

	if (mask.at<uchar>(pCurrent) == GC_PR_FGD) {
		circle(res, pCurrent, 1, pxls, thickness);
		circle(mask, pCurrent, 1, value, thickness);
		Point p;
		for (p.x = pCurrent.x - 1; p.x < pCurrent.x + 2;p.x++) {
			for (p.y = pCurrent.y - 1; p.y < pCurrent.y + 2;p.y++) {
				if (p.x >= 0 && p.y >= 0 && p.x < video.get(CV_CAP_PROP_FRAME_WIDTH) - 1 && p.y < video.get(CV_CAP_PROP_FRAME_HEIGHT) - 1) {
					Vec3b color1 = image->at<Vec3b>(p);
					Vec3b color2 = image->at<Vec3b>(pCurrent);
					Vec3b color3 = image->at<Vec3b>(pCenter);
					bool p_pCurrent = abs(color1[0] - color2[0]) <= 8 && abs(color1[1] - color2[1]) <= 8 && abs(color1[2] - color2[2]) <= 8;
					bool p_pCenter = abs(color1[0] - color3[0]) <= 16 && abs(color1[1] - color3[1]) <= 16 && abs(color1[2] - color3[2]) <= 16;
					if (p_pCurrent && p_pCenter&&mask.at<uchar>(p) == GC_PR_FGD) {
						reLblsInMask(p, pCenter, isFGD);
					}
				}
			}
		}
	}
}



void GCApplication::mouseClick(int event, int x, int y, int flags, void*)
{
	int xMax = video.get(CV_CAP_PROP_FRAME_WIDTH) - 1;
	int yMax = video.get(CV_CAP_PROP_FRAME_HEIGHT) - 1;
	x = x >= 0 ? x : 0;
	x = x <= xMax ? x : xMax;
	y = y >= 0 ? y : 0;
	y = y <= yMax ? y : yMax;
	// TODO add bad args check
	switch (event)
	{
	case EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels
	{
		bool isb = (flags & BGD_KEY) != 0,
			isf = (flags & FGD_KEY) != 0;
		if (rectState != IN_PROCESS && !isb && !isf)
		{
			rectState = IN_PROCESS;
			rect = Rect(x, y, 1, 1);
		}
		if ((isb || isf) )
			lblsState = IN_PROCESS;
	}
	break;
	case EVENT_RBUTTONDOWN: // set GC_PR_BGD(GC_PR_FGD) labels
	{
		bool isb = (flags & BGD_KEY) != 0,
			isf = (flags & FGD_KEY) != 0;
		if ((isb || isf) && rectState == SET)
			prLblsState = IN_PROCESS;
	}
	break;
	case EVENT_LBUTTONUP:
		if (rectState == IN_PROCESS)
		{
			rect = Rect(Point(rect.x, rect.y), Point(x, y));
			rectState = SET;
			setRectInMask();
			showImage();
		}
		if (lblsState == IN_PROCESS)
		{
			if (flags & BGD_KEY)
			{
				reLblsInMask(Point(x, y), Point(x, y), false);
			}
			if (flags & FGD_KEY)
			{
				reLblsInMask(Point(x, y), Point(x, y), true);
			}
			lblsState = SET;
			showImage();
		}
		break;
	case EVENT_RBUTTONUP:
		if (prLblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), true);
			prLblsState = SET;
			showImage();
		}
		break;
	case EVENT_MOUSEMOVE:
		if (rectState == IN_PROCESS)
		{
			rect = Rect(Point(rect.x, rect.y), Point(x, y));
			showImage();
		}
		else if (lblsState == IN_PROCESS)
		{
			if (flags & BGD_KEY)
			{
				reLblsInMask(Point(x, y), Point(x, y), false);
			}
			if (flags & FGD_KEY)
			{
				reLblsInMask(Point(x, y), Point(x, y), true);
			}
			showImage();
		}
		else if (prLblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), true);
			showImage();
		}
		break;
	}
}

GCApplication gcapp;

static void on_mouse(int event, int x, int y, int flags, void* param)
{
	gcapp.mouseClick(event, x, y, flags, param);
}



int main() {
	string openName = "333";
	video.open("E:/Projects/OpenCV/DAVIS-data/image/" + openName + ".avi");
	videowriter.open("E:/Projects/OpenCV/DAVIS-data/image/1output.avi", CV_FOURCC('D', 'I', 'V', 'X'), 5, Size(video.get(CV_CAP_PROP_FRAME_WIDTH), video.get(CV_CAP_PROP_FRAME_HEIGHT)));

	//Mat tureMask = imread("E:/Projects/OpenCV/DAVIS-data/image/00004.png", 0);
	//CAP_PROP_FRAME_COUNTCompatTelRunner
	for (int times = 0; times < 1; times++)
	{
		int key[3] = { 4,13,22 };
		for (int i = 0;i < 27;i++) {
			Mat imgSrc;
			video >> imgSrc;
			imgSrcArr.push_back(imgSrc);
		}

		//for (int i = 0;i < 3;i++) {
		//	string name = "E:/Projects/OpenCV/DAVIS-data/image/mask/" + openName + "/" + to_string(i) + ".bmp";
		//	Mat mask = imread(name, 0);
		//	keyMaskArr.push_back(mask);
		//	imshow("目标", imgSrcArr[0]);//显示结果
		//}

		std::vector<Mat> imgInteArr;
		Mat inteMask, gcappImg;
		bool isInitInte = false;
		for (int i = 0;i < 3;i++) {
			const string winName = "原图像";
			if (isInitInte) {
				gcapp.reset();
				namedWindow(winName, WINDOW_AUTOSIZE);
				imgSrcArr[key[i]].copyTo(gcappImg);
				gcapp.setImageAndWinName(gcappImg, winName);

				Mat img1,img2;
				imgSrcArr[key[i]].copyTo(img1);
				imgSrcArr[key[i-1]].copyTo(img2);
				imgInteArr.push_back(img1);
				imgInteArr.push_back(img2);
				BilateralSimple bil(imgInteArr);
				bil.InitGmms(inteMask);
				bil.run(gcapp.mask);

			}
			else {
				gcapp.reset();
				namedWindow(winName, WINDOW_AUTOSIZE);
				imgSrcArr[key[i]].copyTo(gcappImg);
				gcapp.setImageAndWinName(gcappImg, winName);
			}

			setMouseCallback(winName, on_mouse, 0);
			//gcapp.notSetRect();//取消画框
			gcapp.showImage();
			printf("第%d帧\n", key[i]);
			while (1)
			{
				int t = waitKey();
				char c = (char)t;
				if (c == 'n') {   //键盘输入S实现分割
					isInitInte = true;
					break;
				}
				if (c == 'r') {   //键盘输入S实现分割
					gcapp.reset();
					gcapp.showImage();
				}
			}
			Mat mask;
			medianBlur(gcapp.mask, mask, 5);
			gcapp.mask.copyTo(mask);
			keyMaskArr.push_back(mask);
			gcapp.mask.copyTo(inteMask);
			string name = "E:/Projects/OpenCV/DAVIS-data/image/mask/" + openName + "/" + to_string(i) + ".bmp";
			imwrite(name, mask);
		}

		printf("标记结束\n");
		while (1)
		{
			int t = waitKey();
			if (t == 27) break; //27就是esc,随时生效

			char c = (char)t;
			if (c == 's')   //键盘输入S实现分割
			{
				//imwrite("E:/Projects/OpenCV/DAVIS-data/image/0mask.bmp", gcapp.mask);

				printf("第%d段开始分割\n", times + 1);
				double _time = static_cast<double>(getTickCount());
				Bilateral bilateral(imgSrcArr);
				bilateral.InitGmms(keyMaskArr, key);//gcapp.mask   tureMask
				bilateral.run(maskArr);
				_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
				printf("总用时为%f\n", _time);//显示时间

				for (int t = 0; t < imgSrcArr.size(); t++)
				{
					Mat mask = maskArr[t];
					Mat maskBlur;
					Mat lastImg(maskArr[t].size(), CV_8UC3, cv::Scalar(0, 0, 0));
					medianBlur(mask, maskBlur, 5);
					imgSrcArr[t].copyTo(lastImg, maskBlur);
					//	imshow("目标", lastImg);//显示结果
					//string name = "E:/Projects/OpenCV/DAVIS-data/image/mask/第" + to_string(t + 1) + "帧.bmp";
					//imwrite(name, lastImg);
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
				printf("第%d段分割结束\n", times + 1);

				videowriter.release();
				//break;
			}
		}
	}

	//videowriter.release();
	video.release();
	imgSrcArr.clear();
	cv::destroyAllWindows();
	return 0;
}