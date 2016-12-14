#include "BilateralSimple.h"

BilateralSimple::BilateralSimple(std::vector<Mat> img):
	imgSrcArr(img) {
		initGrid();
	}


BilateralSimple::~BilateralSimple()
{
	for (int i = imgSrcArr.size() - 1;i >= 0;i--) {
		imgSrcArr[i].release();
	}
	imgSrcArr.clear();
	bgModel.release();
	fgModel.release();
	grid.release();
}

void BilateralSimple::initGrid() {
	double _time = static_cast<double>(getTickCount());

	Mat L(6, gridSize, CV_32SC(4), Scalar(0, 0, 0, -1));
	Mat C(6, gridSize, CV_32FC(3), Scalar::all(0));
	grid = L;
	gridColor = C;
	int tSize = imgSrcArr.size();
	int xSize = imgSrcArr[0].rows;
	int ySize = imgSrcArr[0].cols;
	for (int t = 0; t < tSize; t++)
	{
		for (int x = 0; x < xSize; x++)
		{
			//#pragma omp parallel for
			for (int y = 0; y < ySize; y++)
			{
				int tNew = gridSize[0] * t / tSize;
				int xNew = gridSize[1] * x / xSize;
				int yNew = gridSize[2] * y / ySize;
				Vec3b color = (Vec3b)imgSrcArr[t].at<Vec3b>(x, y);
				int rNew = gridSize[3] * color[0] / 256;
				int gNew = gridSize[4] * color[1] / 256;
				int bNew = gridSize[5] * color[2] / 256;
				int point[6] = { tNew,xNew,yNew,rNew,gNew,bNew };
				int count = ++(grid.at<Vec< int, 4 > >(point)[pixSum]);
				Vec3f colorMeans = gridColor.at<Vec3f>(point);
				colorMeans[0] = colorMeans[0] * (count - 1.0) / (count + 0.0) + color[0] / (count + 0.0);
				colorMeans[1] = colorMeans[1] * (count - 1.0) / (count + 0.0) + color[1] / (count + 0.0);
				colorMeans[2] = colorMeans[2] * (count - 1.0) / (count + 0.0) + color[2] / (count + 0.0);
				gridColor.at<Vec3f>(point) = colorMeans;
			}
		}
	}
	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("����grid��ʱ%f\n", _time);//��ʾʱ��
}

void BilateralSimple::InitGmms(Mat& mask)
{
	double _time = static_cast<double>(getTickCount());//��ʱ

	int tSize = imgSrcArr.size();
	int xSize = imgSrcArr[0].rows;
	int ySize = imgSrcArr[0].cols;
	int point[6] = { 0,0,0,0,0,0 };

		for (int x = 0; x < xSize; x++)
		{
			for (int y = 0; y < ySize; y++)
			{
				if (mask.at<uchar>(x, y) == GC_BGD) {
					getGridPoint(0, Point(x, y), point, tSize, xSize, ySize);
					grid.at<Vec< int, 4 > >(point)[bgdSum] += 5;
					if (point[0] > 0) {
						int pointN[6] = { point[0] - 1,point[1],point[2],point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[1] > 0) {
						int pointN[6] = { point[0],point[1] - 1,point[2],point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[2] > 0) {
						int pointN[6] = { point[0],point[1],point[2] - 1,point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[3] > 0) {
						int pointN[6] = { point[0] ,point[1],point[2],point[3] - 1,point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[4] > 0) {
						int pointN[6] = { point[0],point[1] ,point[2],point[3],point[4] - 1,point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[5] > 0) {
						int pointN[6] = { point[0],point[1],point[2] ,point[3],point[4],point[5] - 1 };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[0] < gridSize[0] - 1) {
						int pointN[6] = { point[0] + 1,point[1],point[2],point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[1] < gridSize[1] - 1) {
						int pointN[6] = { point[0],point[1] + 1,point[2],point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[2] < gridSize[2] - 1) {
						int pointN[6] = { point[0],point[1],point[2] + 1,point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[3] < gridSize[3] - 1) {
						int pointN[6] = { point[0] ,point[1],point[2],point[3] + 1,point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[4] < gridSize[4] - 1) {
						int pointN[6] = { point[0],point[1],point[2],point[3],point[4] + 1,point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[5] < gridSize[5] - 1) {
						int pointN[6] = { point[0],point[1],point[2],point[3],point[4],point[5] + 1 };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
				}
				else if (mask.at<uchar>(x, y) == GC_FGD/*GC_FGD*/) {
					getGridPoint(0, Point(x, y), point, tSize, xSize, ySize);
					grid.at<Vec< int, 4 > >(point)[fgdSum] += 5;
					if (point[0] > 0) {
						int pointN[6] = { point[0] - 1,point[1],point[2],point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[1] > 0) {
						int pointN[6] = { point[0],point[1] - 1,point[2],point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[2] > 0) {
						int pointN[6] = { point[0],point[1],point[2] - 1,point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[3] > 0) {
						int pointN[6] = { point[0] ,point[1],point[2],point[3] - 1,point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[4] > 0) {
						int pointN[6] = { point[0],point[1] ,point[2],point[3],point[4] - 1,point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[5] > 0) {
						int pointN[6] = { point[0],point[1],point[2] ,point[3],point[4],point[5] - 1 };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[0] < gridSize[0] - 1) {
						int pointN[6] = { point[0] + 1,point[1],point[2],point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[1] < gridSize[1] - 1) {
						int pointN[6] = { point[0],point[1] + 1,point[2],point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[2] < gridSize[2] - 1) {
						int pointN[6] = { point[0],point[1],point[2] + 1,point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[3] < gridSize[3] - 1) {
						int pointN[6] = { point[0] ,point[1],point[2],point[3] + 1,point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[4] < gridSize[4] - 1) {
						int pointN[6] = { point[0],point[1] ,point[2],point[3],point[4] + 1,point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[5] < gridSize[5] - 1) {
						int pointN[6] = { point[0],point[1],point[2] ,point[3],point[4],point[5] + 1 };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
				}
				else if (mask.at<uchar>(x, y) == GC_PR_FGD/*GC_FGD*/) {
					getGridPoint(0, Point(x, y), point, tSize, xSize, ySize);
					grid.at<Vec< int, 4 > >(point)[fgdSum] += 2;
					if (point[0] > 0) {
						int pointN[6] = { point[0] - 1,point[1],point[2],point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[1] > 0) {
						int pointN[6] = { point[0],point[1] - 1,point[2],point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[2] > 0) {
						int pointN[6] = { point[0],point[1],point[2] - 1,point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[3] > 0) {
						int pointN[6] = { point[0] ,point[1],point[2],point[3] - 1,point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[4] > 0) {
						int pointN[6] = { point[0],point[1] ,point[2],point[3],point[4] - 1,point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[5] > 0) {
						int pointN[6] = { point[0],point[1],point[2] ,point[3],point[4],point[5] - 1 };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[0] < gridSize[0] - 1) {
						int pointN[6] = { point[0] + 1,point[1],point[2],point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[1] < gridSize[1] - 1) {
						int pointN[6] = { point[0],point[1] + 1,point[2],point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[2] < gridSize[2] - 1) {
						int pointN[6] = { point[0],point[1],point[2] + 1,point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[3] < gridSize[3] - 1) {
						int pointN[6] = { point[0] ,point[1],point[2],point[3] + 1,point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[4] < gridSize[4] - 1) {
						int pointN[6] = { point[0],point[1] ,point[2],point[3],point[4] + 1,point[5] };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
					if (point[5] < gridSize[5] - 1) {
						int pointN[6] = { point[0],point[1],point[2] ,point[3],point[4],point[5] + 1 };
						grid.at<Vec< int, 4 > >(pointN)[fgdSum] += 1;
					}
				}
				else if (mask.at<uchar>(x, y) == GC_PR_BGD/*GC_FGD*/) {
					getGridPoint(0, Point(x, y), point, tSize, xSize, ySize);
					grid.at<Vec< int, 4 > >(point)[bgdSum] += 2;
					if (point[0] > 0) {
						int pointN[6] = { point[0] - 1,point[1],point[2],point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[1] > 0) {
						int pointN[6] = { point[0],point[1] - 1,point[2],point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[2] > 0) {
						int pointN[6] = { point[0],point[1],point[2] - 1,point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[3] > 0) {
						int pointN[6] = { point[0] ,point[1],point[2],point[3] - 1,point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[4] > 0) {
						int pointN[6] = { point[0],point[1] ,point[2],point[3],point[4] - 1,point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[5] > 0) {
						int pointN[6] = { point[0],point[1],point[2] ,point[3],point[4],point[5] - 1 };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[0] < gridSize[0] - 1) {
						int pointN[6] = { point[0] + 1,point[1],point[2],point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[1] < gridSize[1] - 1) {
						int pointN[6] = { point[0],point[1] + 1,point[2],point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[2] < gridSize[2] - 1) {
						int pointN[6] = { point[0],point[1],point[2] + 1,point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[3] < gridSize[3] - 1) {
						int pointN[6] = { point[0] ,point[1],point[2],point[3] + 1,point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[4] < gridSize[4] - 1) {
						int pointN[6] = { point[0],point[1],point[2],point[3],point[4] + 1,point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[5] < gridSize[5] - 1) {
						int pointN[6] = { point[0],point[1],point[2],point[3],point[4],point[5] + 1 };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
				}
			}
		}

	std::vector<Vec3f> bgdSamples;    //�ӱ�����洢������ɫ
	std::vector<Vec3f> fgdSamples;    //��ǰ����洢ǰ����ɫ
	std::vector<double> bgdWeight;    //�ӱ�����洢����Ȩֵ
	std::vector<double> fgdWeight;    //��ǰ����洢ǰ��Ȩֵ

	for (int t = 0; t < gridSize[0]; t++) {
		for (int x = 0; x < gridSize[1]; x++) {
			for (int y = 0; y < gridSize[2]; y++) {
				for (int r = 0; r < gridSize[3]; r++) {
					for (int g = 0; g < gridSize[4]; g++) {
						for (int b = 0; b < gridSize[5]; b++) {
							int point[6] = { t,x,y,r,g,b };
							int pixCount = grid.at<Vec< int, 4 > >(point)[pixSum];
							if (pixCount > 0) {
								int bgdcount = grid.at<Vec< int, 4 > >(point)[bgdSum];
								int fgdcount = grid.at<Vec< int, 4 > >(point)[fgdSum];
								
								if (bgdcount > pixCount) {
									Vec3f color = gridColor.at<Vec3f>(point);
									bgdSamples.push_back(color);
									bgdWeight.push_back(bgdcount);
								}
								if (fgdcount > pixCount) {
									Vec3f color = gridColor.at<Vec3f>(point);
									fgdSamples.push_back(color);
									fgdWeight.push_back(fgdcount);
								}
							}
						}
					}
				}
			}
		}
	}


		GMM bgdGMM(bgModel), fgdGMM(fgModel);
		const int kMeansItCount = 10;  //��������  
		const int kMeansType = KMEANS_PP_CENTERS; //Use kmeans++ center initialization by Arthur and Vassilvitskii  
		Mat bgdLabels, fgdLabels; //��¼������ǰ����������������ÿ�����ض�ӦGMM���ĸ���˹ģ��
								  //kmeans���з���
		Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
		kmeans(_bgdSamples, GMM::componentsCount, bgdLabels,
			TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);
		Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
		kmeans(_fgdSamples, GMM::componentsCount, fgdLabels,
			TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);

		//�����Ľ������ѵ��GMMs����ʼ����
		bgdGMM.initLearning();
		for (int i = 0; i < (int)bgdSamples.size(); i++)
			bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i], bgdWeight[i]);
		bgdGMM.endLearning();
		fgdGMM.initLearning();
		for (int i = 0; i < (int)fgdSamples.size(); i++)
			fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i], fgdWeight[i]);
		fgdGMM.endLearning();
		for (int times = 0; times < 2; times++)
		{
			//ѵ��GMMsģ��
			for (int i = 0; i < (int)bgdSamples.size(); i++) {
				Vec3d color = bgdSamples[i];
				bgdLabels.at<int>(i, 0) = bgdGMM.whichComponent(color);
			}

			for (int i = 0; i < (int)fgdSamples.size(); i++) {
				Vec3d color = fgdSamples[i];
				fgdLabels.at<int>(i, 0) = fgdGMM.whichComponent(color);
			}

			bgdGMM.initLearning();
			for (int i = 0; i < (int)bgdSamples.size(); i++)
				bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i], bgdWeight[i]);
			bgdGMM.endLearning();
			fgdGMM.initLearning();
			for (int i = 0; i < (int)fgdSamples.size(); i++)
				fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i], fgdWeight[i]);
			fgdGMM.endLearning();
		}

		std::vector<Vec3f> unSamples;    //��������
		for (int i = 0; i < (int)bgdSamples.size(); i++) {
			Vec3d color = bgdSamples[i];
			if (bgdGMM(color) < fgdGMM(color)) {
				unSamples.push_back(color);
			}
		}
		for (int i = 0; i < (int)fgdSamples.size(); i++) {
			Vec3d color = fgdSamples[i];
			if (bgdGMM(color) > fgdGMM(color)) {
				unSamples.push_back(color);
			}
		}
		GMM unGMM(unModel);
		Mat unLabels;
		Mat _unSamples((int)unSamples.size(), 3, CV_32FC1, &unSamples[0][0]);
		kmeans(_unSamples, GMM::componentsCount, unLabels,
			TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);
		unGMM.initLearning();
		for (int i = 0; i < (int)unSamples.size(); i++)
			unGMM.addSample(unLabels.at<int>(i, 0), unSamples[i], 1);
		unGMM.endLearning();
		for (int times = 0; times < 2; times++)
		{
			//ѵ��GMMsģ��
			for (int i = 0; i < (int)unSamples.size(); i++) {
				Vec3d color = unSamples[i];
				unLabels.at<int>(i, 0) = unGMM.whichComponent(color);
			}
			unGMM.initLearning();
			for (int i = 0; i < (int)unSamples.size(); i++)
				unGMM.addSample(unLabels.at<int>(i, 0), unSamples[i], 1);
			unGMM.endLearning();
		}


	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("��˹��ģ��ʱ%f\n", _time);//��ʾʱ��
}


static double calcBeta(const Mat& img)
{
	double beta = 0;
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			//�����ĸ��������������صĲ��Ҳ����ŷʽ�������˵���׷���  
			//�����������ض�����󣬾��൱�ڼ������������ز��ˣ�  
			Vec3d color = img.at<Vec3b>(y, x);
			if (x > 0) // left  >0���ж���Ϊ�˱�����ͼ��߽��ʱ�򻹼��㣬����Խ��  
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x - 1);
				beta += diff.dot(diff);  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�  
			}
			if (y > 0 && x > 0) // upleft  
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x - 1);
				beta += diff.dot(diff);
			}
			if (y > 0) // up  
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x);
				beta += diff.dot(diff);
			}
			if (y > 0 && x < img.cols - 1) // upright  
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x + 1);
				beta += diff.dot(diff);
			}
		}
	}
	if (beta <= std::numeric_limits<double>::epsilon())
		beta = 0;
	else
		beta = 1.f / (2 * beta / (4 * img.cols*img.rows - 3 * img.cols - 3 * img.rows + 2)); //���Ĺ�ʽ��5��  

	return beta;
}

void BilateralSimple::constructGCGraph(GCGraph<double>& graph) {
	double _time = static_cast<double>(getTickCount());

	//double bata = calcBeta(imgSrcArr[0]);
	double bata = 0.01;
	int vtxCount = calculateVtxCount();  //��������ÿһ��������һ������  
	int edgeCount = 2 * 256 * vtxCount;  //��������Ҫ����ͼ�߽�ıߵ�ȱʧ
	graph.create(vtxCount, edgeCount);
	GMM bgdGMM(bgModel), fgdGMM(fgModel),unGMM(unModel);
	for (int t = 0; t < gridSize[0]; t++) {
		for (int x = 0; x < gridSize[1]; x++) {
			for (int y = 0; y < gridSize[2]; y++) {
				for (int r = 0; r < gridSize[3]; r++) {
					for (int g = 0; g < gridSize[4]; g++) {
						for (int b = 0; b < gridSize[5]; b++) {
							int point[6] = { t,x,y,r,g,b };
							int pixCount = grid.at<Vec< int, 4 > >(point)[pixSum];
							if (pixCount > 0) {
								int vtxIdx = graph.addVtx();//�������ص�ӳ��ͼӶ���

															//������
								grid.at<Vec< int, 4 > >(point)[vIdx] = vtxIdx;

								Vec3f color = gridColor.at<Vec3f>(point);
								double fromSource, toSink;

								double fSum = grid.at<Vec< int, 4 > >(point)[fgdSum];
								double bSum = grid.at<Vec< int, 4 > >(point)[bgdSum];
								//�ۺϷ���
								if ((bSum > 4 * pixCount) && fSum == 0) {
									fromSource = 0;
									toSink = 9999;
								}
								else if (bSum == 0 && (fSum > 4 * pixCount)) {
									fromSource = 9999;
									toSink = 0;
								}
								else {
									double bgd = bgdGMM(color);
									double fgd = fgdGMM(color);
									double un = unGMM(color);
									double weight = un / (bgd + fgd + un);
									/*double weight = 0.5;*/
									fromSource = (-log(bgd / (bgd + fgd))*(1 - weight) - log((bSum + 1) / (fSum + bSum + 1))*weight)*sqrt(pixCount);
									toSink = (-log(fgd / (bgd + fgd))*(1 - weight) - log((fSum + 1) / (fSum + bSum + 1))*weight)*sqrt(pixCount);
									/*fromSource = (-log(bgd / (bgd + fgd)))*sqrt(pixCount);
									toSink = (-log(fgd / (bgd + fgd)))*sqrt(pixCount);*/
									int a = 1;
								}

								//ԭ����
								/*fromSource =  (toSinkSum + 2);
								toSink =  (fromSourceSum + 2);*/


								graph.addTermWeights(vtxIdx, fromSource, toSink);


								//ƽ����
								if (t > 0) {
									int pointN[6] = { t - 1,x,y,r,g,b };
									if (grid.at<Vec< int, 4 > >(pointN)[pixSum] > 0) {
										double num = sqrt(grid.at<Vec< int, 4 > >(point)[pixSum] * grid.at<Vec< int, 4 > >(pointN)[pixSum] + 1);
										Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
										double e = exp(-bata*diff.dot(diff));  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�
										double w = 1.0 * e * sqrt(num);
										graph.addEdges(vtxIdx, grid.at<Vec< int, 4 > >(pointN)[vIdx], w, w);
									}
								}
								if (x > 0) {
									int pointN[6] = { t,x - 1,y,r,g,b };
									if (grid.at<Vec< int, 4 >>(pointN)[pixSum] > 0) {
										double num = sqrt(grid.at<Vec< int, 4 > >(point)[pixSum] * grid.at<Vec< int, 4 > >(pointN)[pixSum] + 1);
										Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
										double e = exp(-bata*diff.dot(diff));  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�
										double w = 1.0 * e * sqrt(num);
										graph.addEdges(vtxIdx, grid.at<Vec< int, 4 > >(pointN)[vIdx], w, w);
									}
								}
								if (y > 0) {
									int pointN[6] = { t,x,y - 1,r,g,b };
									if (grid.at<Vec< int, 4 > >(pointN)[pixSum] > 0) {
										double num = sqrt(grid.at<Vec< int, 4 > >(point)[pixSum] * grid.at<Vec< int, 4 > >(pointN)[pixSum] + 1);
										Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
										double e = exp(-bata*diff.dot(diff));  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�
										double w = 1.0 * e * sqrt(num);
										graph.addEdges(vtxIdx, grid.at<Vec< int, 4 > >(pointN)[vIdx], w, w);
									}
								}
								if (r > 0) {
									int pointN[6] = { t,x,y,r - 1,g,b };
									if (grid.at<Vec< int, 4 > >(pointN)[pixSum] > 0) {
										double num = sqrt(grid.at<Vec< int, 4 > >(point)[pixSum] * grid.at<Vec< int, 4 > >(pointN)[pixSum] + 1);
										Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
										double e = exp(-bata*diff.dot(diff));  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�
										double w = 1.0 * e * sqrt(num);
										graph.addEdges(vtxIdx, grid.at<Vec< int, 4 > >(pointN)[vIdx], w, w);
									}
								}
								if (g > 0) {
									int pointN[6] = { t,x,y,r,g - 1,b };
									if (grid.at<Vec< int, 4 > >(pointN)[pixSum] > 0) {
										double num = sqrt(grid.at<Vec< int, 4 > >(point)[pixSum] * grid.at<Vec< int, 4 > >(pointN)[pixSum] + 1);
										Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
										double e = exp(-bata*diff.dot(diff));  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�
										double w = 1.0 * e * sqrt(num);
										graph.addEdges(vtxIdx, grid.at<Vec< int, 4 > >(pointN)[vIdx], w, w);
									}
								}
								if (b > 0) {
									int pointN[6] = { t,x,y,r,g,b - 1 };
									if (grid.at<Vec< int, 4 > >(pointN)[pixSum] > 0) {
										double num = sqrt(grid.at<Vec< int, 4 > >(point)[pixSum] * grid.at<Vec< int, 4 > >(pointN)[pixSum] + 1);
										Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
										double e = exp(-bata*diff.dot(diff));  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�
										double w = 1.0 * e * sqrt(num);
										graph.addEdges(vtxIdx, grid.at<Vec< int, 4 > >(pointN)[vIdx], w, w);
									}
								}
							}
						}
					}
				}

			}
		}
	}

	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("ͼ�ͼ��ʱ %f\n", _time);//��ʾʱ��
}


int BilateralSimple::calculateVtxCount() {
	int count = 0;
	for (int t = 0; t < gridSize[0]; t++)
	{
		for (int x = 0; x < gridSize[1]; x++)
		{
			for (int y = 0; y < gridSize[2]; y++)
			{
				for (int r = 0; r < gridSize[3]; r++)
				{
					for (int g = 0; g < gridSize[4]; g++)
					{
						for (int b = 0; b < gridSize[5]; b++)
						{
							int point[6] = { t,x,y,r,g,b };
							if (grid.at<Vec< int, 4 > >(point)[pixSum] > 0) {
								count++;
							}
						}
					}
				}
			}
		}
	}
	return count;
}

void BilateralSimple::estimateSegmentation(GCGraph<double>& graph, Mat& mask) {
	double _time = static_cast<double>(getTickCount());
	graph.maxFlow();//�����ͼ��
	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("ͼ��ָ���ʱ %f\n", _time);//��ʾʱ��

	double _time2 = static_cast<double>(getTickCount());
	int xSize = imgSrcArr[0].rows;
	int ySize = imgSrcArr[0].cols;
		for (int y = 0; y < ySize; y++)
		{
#pragma omp parallel for
			for (int x = 0; x < xSize; x++)
			{
				Point p(x, y);
				int point[6] = { 0,0,0,0,0,0 };
				getGridPoint(1, p, point, 2, xSize, ySize);
				int vertex = grid.at<Vec< int, 4 > >(point)[vIdx];
				if (graph.inSourceSegment(vertex))
					mask.at<uchar>(p.x, p.y) = GC_PR_FGD;
				else
					mask.at<uchar>(p.x, p.y) = GC_PR_BGD;
			}
		}

	_time2 = (static_cast<double>(getTickCount()) - _time2) / getTickFrequency();
	printf("grid�������mask��ʱ %f\n", _time2);//��ʾʱ��
}

void BilateralSimple::getGridPoint(int index, const Point p, int *point, int tSize, int xSize, int ySize) {
	point[0] = gridSize[0] * index / tSize;
	point[1] = gridSize[1] * p.x / xSize;
	point[2] = gridSize[2] * p.y / ySize;
	Vec3b color = (Vec3b)imgSrcArr[index].at<Vec3b>(p.x, p.y);
	point[3] = gridSize[3] * color[0] / 256;
	point[4] = gridSize[4] * color[1] / 256;
	point[5] = gridSize[5] * color[2] / 256;
}

void BilateralSimple::getGridPoint(int index, const Point p, std::vector<int>& point, int tSize, int xSize, int ySize) {
	point[0] = gridSize[0] * index / tSize;
	point[1] = gridSize[1] * p.y / xSize;
	point[2] = gridSize[2] * p.x / ySize;//x,y����������p���������µ����⡣
	Vec3b color = (Vec3b)imgSrcArr[index].at<Vec3b>(p);
	point[3] = gridSize[3] * color[0] / 256;
	point[4] = gridSize[4] * color[1] / 256;
	point[5] = gridSize[5] * color[2] / 256;
}


void BilateralSimple::run(Mat& mask) {
	mask.create(imgSrcArr[0].rows, imgSrcArr[0].cols, CV_8UC1);

	GCGraph<double> graph;//ͼ��
	constructGCGraph(graph);
	estimateSegmentation(graph, mask);
}