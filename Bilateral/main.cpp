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
enum GrabCut {
	GC_INIT = 10
};


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
		mask.setTo(Scalar::all(GC_INIT));
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

void GCApplication::reLblsInMask(Point pC, Point pCenter, bool isFGD)
{
	uchar value = isFGD ? GC_FGD : GC_BGD;
	uchar canDoLbl = isFGD ? GC_PR_BGD : GC_PR_FGD;
	Scalar pxls = isFGD ? RED : BLUE;

	vector<Point> pointList;
	pointList.push_back(pC);

	while (!pointList.empty())
	{
		Point pCurrent = pointList.back();
		pointList.pop_back();
		if (mask.at<uchar>(pCurrent) == GC_INIT || mask.at<uchar>(pCurrent) == canDoLbl) {
			circle(res, pCurrent, 1, pxls, thickness);
			circle(mask, pCurrent, 1, value, thickness);
			Point p;
			for (p.x = pCurrent.x - 1; p.x < pCurrent.x + 2;p.x++) {
				for (p.y = pCurrent.y - 1; p.y < pCurrent.y + 2;p.y++) {
					if (p.x >= 0 && p.y >= 0 && p.x < image->cols - 1 && p.y < image->rows - 1) {
						Vec3b color1 = image->at<Vec3b>(p);
						Vec3b color2 = image->at<Vec3b>(pCurrent);
						Vec3b color3 = image->at<Vec3b>(pCenter);
						Vec3d diff12 = (Vec3d)color1 - (Vec3d)color2;
						Vec3d diff13 = (Vec3d)color1 - (Vec3d)color3;						
						bool p_pCurrent = diff12.dot(diff12) <= 128;
						bool p_pCenter = diff13.dot(diff13) <= 256;
						if (p_pCurrent && p_pCenter && (mask.at<uchar>(p) == GC_INIT || mask.at<uchar>(p) == canDoLbl)) {
							Point ptemp = p;
							pointList.push_back(ptemp);
						}
					}
				}
			}
		}
	}
}

void GCApplication::mouseClick(int event, int x, int y, int flags, void*)
{
	int xMax = image->cols - 1;
	int yMax = image->rows - 1;
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

void blurBySlic(Mat &imgSrc,Mat &mask) {
	Mat lableMat;
	SLIC slic;
	int numSuperpixels = slic.GenerateSuperpixels(imgSrc, 1500);
	slic.GetLabelInMat(lableMat);
	double *fLabel = new double[numSuperpixels];
	double *bLabel = new double[numSuperpixels];
	for (int i = 0;i < numSuperpixels;i++) {
		fLabel[i] = 0;bLabel[i] = 0;
	}
	for (int x = 0;x < mask.rows;x++) {
		for (int y = 0;y < mask.cols;y++) {
			if (mask.at<uchar>(x, y) == 1) {
				fLabel[lableMat.at<int>(x, y)]++;
			}
			else {
				bLabel[lableMat.at<int>(x, y)]++;
			}
		}
	}
	for (int i = 0;i < numSuperpixels;i++) {
		double sum = fLabel[i] + bLabel[i];
		fLabel[i] = fLabel[i] / sum;
		bLabel[i] = bLabel[i] / sum;
	}

	for (int x = 0;x < mask.rows;x++) {
		for (int y = 0;y < mask.cols;y++) {
			if (bLabel[lableMat.at<int>(x, y)]>0.9) {
				mask.at<uchar>(x, y) = 0;
			}else if (fLabel[lableMat.at<int>(x, y)]>0.9) {
				mask.at<uchar>(x, y) = 1;
			}
		}
	}

	delete[] fLabel;
	delete[] bLabel;
}

static void interact(string openName, int* key,int num)
{
	std::vector<Mat> imgInteArr;
	Mat inteMask, gcappImg, maskTemp;//inteMask当前帧的分割结果，maskTemp用来滤波和暗化背景，gcappImg交互显示的图片。
	bool isInitInte = false;
	for (int i = 0;i < num;i++) {
		const string winName = "原图像";
		if (isInitInte) {
			gcapp.reset();
			namedWindow(winName, WINDOW_AUTOSIZE);
			imgSrcArr[key[i]].copyTo(gcappImg);
			gcapp.setImageAndWinName(gcappImg, winName);

			Mat img1,img2;//前一帧与当前帧的图片
			imgSrcArr[key[i-1]].copyTo(img1);
			imgSrcArr[key[i]].copyTo(img2);
			imgInteArr.push_back(img1);
			imgInteArr.push_back(img2);
			BilateralSimple bil(imgInteArr);
			bil.InitGmms(inteMask);
			bil.run(inteMask);
			inteMask.copyTo(maskTemp);
			medianBlur(maskTemp, inteMask, 3);
			inteMask.copyTo(maskTemp);
			inteMask = inteMask & 1;

			Mat img3(img1.size(), CV_8UC3, cv::Scalar(0, 0, 0));
			img2.copyTo(img3, inteMask);

			addWeighted(img2, 0.4, img3, 0.6,0, gcappImg);
			gcapp.reset();
			maskTemp.copyTo(gcapp.mask);

			imgInteArr.clear();

		}
		else {
			gcapp.reset();
			namedWindow(winName, WINDOW_AUTOSIZE);
			imgSrcArr[key[i]].copyTo(gcappImg);
			gcapp.setImageAndWinName(gcappImg, winName);
		}

		setMouseCallback(winName, on_mouse, 0);
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
				if (isInitInte) {
					maskTemp.copyTo(gcapp.mask);
				}
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
}


int main() {
	string openName = "333";
	video.open("E:/Projects/OpenCV/DAVIS-data/image/" + openName + ".avi");
	videowriter.open("E:/Projects/OpenCV/DAVIS-data/image/1output.avi", CV_FOURCC('D', 'I', 'V', 'X'), 5, Size(video.get(CV_CAP_PROP_FRAME_WIDTH), video.get(CV_CAP_PROP_FRAME_HEIGHT)));
	//Mat tureMask = imread("E:/Projects/OpenCV/DAVIS-data/image/00004.png", 0);
	
	/*for (int i = 0;i < 240;i++) {
		Mat imgSrc;
		video.read(imgSrc);
	}*/

	for (int times = 0; times < 1; times++)
	{
		int key[8] = { 2,22,42,62};
		for (int i = 0;i < 65;i++) {
			Mat imgSrc;
			if (video.read(imgSrc)) {
				imgSrcArr.push_back(imgSrc);
			}
			else {
				imgSrcArr[i - 1].copyTo(imgSrc);
				imgSrcArr.push_back(imgSrc);
			}
			
		}

		for (int i = 0;i < 4;i++) {
			string name = "E:/Projects/OpenCV/DAVIS-data/image/mask/" + openName + "/" + to_string(i) + ".bmp";
			Mat mask = imread(name, 0);
			keyMaskArr.push_back(mask);
			imshow("目标", imgSrcArr[0]);//显示结果
		}

		//interact(openName, key,4);

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
				printf("分割用时为%f\n", _time);//显示时间
				double _time1 = static_cast<double>(getTickCount());

				/*#pragma omp parallel for
				for (int t = 0; t < imgSrcArr.size(); t++)
				{
					blurBySlic(imgSrcArr[t], maskArr[t]);
				}*/

				for (int t = 0; t < imgSrcArr.size(); t++)
				{
					Mat mask = maskArr[t];					
					Mat lastImg(maskArr[t].size(), CV_8UC3, cv::Scalar(0, 0, 0));
					Mat maskBlur;
					medianBlur(mask, maskBlur, 5);
					imgSrcArr[t].copyTo(lastImg, maskBlur);

					string name = "E:/Projects/OpenCV/DAVIS-data/image/output/01/result第" + to_string(t + 1) + "帧.bmp";
					imwrite(name, lastImg);

					//Mat img3(imgSrcArr[t].size(), CV_8UC3, cv::Scalar(0, 0, 0));
					//imgSrcArr[t].copyTo(img3, maskBlur);
					//addWeighted(imgSrcArr[t], 0.3, img3, 0.7, 0, lastImg);

					videowriter << lastImg;
				}
				_time1 = (static_cast<double>(getTickCount()) - _time1) / getTickFrequency();
				printf("滤波用时为%f\n", _time1);//显示时间

				std::vector<Mat>  gmmProMaskArr, keyProMaskArr, totalProMaskArr;
				bilateral.getGmmProMask(gmmProMaskArr);
				bilateral.getKeyProMask(keyProMaskArr);
				bilateral.getTotalProMask(totalProMaskArr);
				for (int t = 0; t < imgSrcArr.size(); t++)
				{					
					string gName = "E:/Projects/OpenCV/DAVIS-data/image/output/01/gmmPro第" + to_string(t + 1) + "帧.bmp";
					imwrite(gName, gmmProMaskArr[t]);
					string kName = "E:/Projects/OpenCV/DAVIS-data/image/output/01/keyPro第" + to_string(t + 1) + "帧.bmp";
					imwrite(kName, keyProMaskArr[t]);
					string tName = "E:/Projects/OpenCV/DAVIS-data/image/output/01/totalPro第" + to_string(t + 1) + "帧.bmp";
					imwrite(tName, totalProMaskArr[t]);
				}

				//清除容器，释放内存
				for (int i = imgSrcArr.size() - 1;i >= 0;i--) {
					imgSrcArr[i].release();
				}
				imgSrcArr.clear();
				for (int i = maskArr.size() - 1;i >= 0;i--) {
					maskArr[i].release();
					gmmProMaskArr[i].release(); keyProMaskArr[i].release(); totalProMaskArr[i].release();;
				}
				maskArr.clear();gmmProMaskArr.clear(); keyProMaskArr.clear(); totalProMaskArr.clear();
				printf("第%d段分割结束\n", times + 1);
				
				//videowriter.release();
				//break;
			}
		}
	}

	videowriter.release();
	video.release();
	imgSrcArr.clear();
	cv::destroyAllWindows();
	return 0;
}