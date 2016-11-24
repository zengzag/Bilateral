#include "Bilateral.h"
#include <iostream>

Bilateral::Bilateral(std::vector<Mat> img) :
	imgSrcArr(img) {
	initGrid();
}

Bilateral::~Bilateral()
{
	for (int i = imgSrcArr.size() - 1;i >= 0;i--) {
		imgSrcArr[i].release();
	}
	imgSrcArr.clear();
	bgModel.release();
	fgModel.release();
	grid.release();
}

void Bilateral::InitGmms(Mat& mask, int index)
{
	keyMask = mask;
	double _time = static_cast<double>(getTickCount());//��ʱ

	std::vector<Vec3f> bgdSamples;    //�ӱ�����洢������ɫ
	std::vector<Vec3f> fgdSamples;    //��ǰ����洢ǰ����ɫ

	int tSize = imgSrcArr.size();
	int xSize = imgSrcArr[0].rows;
	int ySize = imgSrcArr[0].cols;
	int point[6] = { 0,0,0,0,0,0 };

	for (int x = 0; x < xSize; x++)
	{
		for (int y = 0; y < ySize; y++)
		{

			if (mask.at<uchar>(x, y) == GC_BGD) {
				Vec3f color = (Vec3f)imgSrcArr[index].at<Vec3b>(x, y);
				bgdSamples.push_back(color);
				getGridPoint(index, Point(x, y), point, tSize, xSize, ySize);
				if (grid.at<Vec< int, 3 > >(point)[2] != GC_FGD)//ͬһ��������2�����ϱ��ʱѡ����
					grid.at<Vec< int, 3 > >(point)[2] = GC_BGD;
			}
			else if (mask.at<uchar>(x, y) == GC_FGD) {
				Vec3f color = (Vec3f)imgSrcArr[index].at<Vec3b>(x, y);
				fgdSamples.push_back(color);
				getGridPoint(index, Point(x, y), point, tSize, xSize, ySize);
				grid.at<Vec< int, 3 > >(point)[2] = GC_FGD;
			}
			else if (mask.at<uchar>(x, y) == GC_PR_FGD) {
				Vec3f color = (Vec3f)imgSrcArr[index].at<Vec3b>(x, y);
				fgdSamples.push_back(color);
				getGridPoint(index, Point(x, y), point, tSize, xSize, ySize);
				if (grid.at<Vec< int, 3 > >(point)[2] == -1)
					grid.at<Vec< int, 3 > >(point)[2] = GC_PR_FGD;
			}
			else if (mask.at<uchar>(x, y) == GC_PR_BGD) {
				Vec3f color = (Vec3f)imgSrcArr[index].at<Vec3b>(x, y);
				bgdSamples.push_back(color);
				getGridPoint(index, Point(x, y), point, tSize, xSize, ySize);
				if (grid.at<Vec< int, 3 > >(point)[2] == -1)
					grid.at<Vec< int, 3 > >(point)[2] = GC_PR_BGD;
			}
		}
	}

	//��˹ģ�ͽ���
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
		bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	bgdGMM.endLearning();
	fgdGMM.initLearning();
	for (int i = 0; i < (int)fgdSamples.size(); i++)
		fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
	fgdGMM.endLearning();

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
		bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	bgdGMM.endLearning();
	fgdGMM.initLearning();
	for (int i = 0; i < (int)fgdSamples.size(); i++)
		fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
	fgdGMM.endLearning();

	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("��˹��ģ��ʱ%f\n", _time);//��ʾʱ��
}


void Bilateral::initGrid() {
	double _time = static_cast<double>(getTickCount());

	Mat L(6, gridSize, CV_32SC(3), Scalar::all(-1));
	grid = L;
	int tSize = imgSrcArr.size();
	int xSize = imgSrcArr[0].rows;
	int ySize = imgSrcArr[0].cols;
	for (int t = 0; t < tSize; t++)
	{
		for (int x = 0; x < xSize; x++)
		{
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
				grid.at<Vec< int, 3 > >(point)[0] += 1;
			}
		}
	}
	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("����grid��ʱ%f\n", _time);//��ʾʱ��
}

void Bilateral::constructGCGraph(const GMM& bgdGMM, const GMM& fgdGMM, GCGraph<double>& graph) {
	double _time = static_cast<double>(getTickCount());

	int vtxCount = calculateVtxCount();  //��������ÿһ��������һ������  
	int edgeCount = 2 * 6 * vtxCount;  //��������Ҫ����ͼ�߽�ıߵ�ȱʧ
	graph.create(vtxCount, edgeCount);

	for (int t = 0; t < gridSize[0]; t++) {
		for (int x = 0; x < gridSize[1]; x++) {
			for (int y = 0; y < gridSize[2]; y++) {
				for (int r = 0; r < gridSize[3]; r++) {
					for (int g = 0; g < gridSize[4]; g++) {
						for (int b = 0; b < gridSize[5]; b++) {
							int point[6] = { t,x,y,r,g,b };

							if (grid.at<Vec< int, 3 > >(point)[0] != -1) {
								int vtxIdx = graph.addVtx();//�������ص�ӳ��ͼӶ���

								//������
								grid.at<Vec< int, 3 > >(point)[1] = vtxIdx;
								Vec3b color;//����grid�ж����Ӧ����ɫ
								color[0] = (r * 256 + 256 / 2) / gridSize[3];//���256/2��Ϊ�˰���ɫ�Ƶ���������
								color[1] = (g * 256 + 256 / 2) / gridSize[4];
								color[2] = (b * 256 + 256 / 2) / gridSize[5];
								double fromSource, toSink;

								if (grid.at<Vec< int, 3 > >(point)[2] == GC_FGD) {
									fromSource = 9999;
									toSink = 0;
								}
								else if (grid.at<Vec< int, 3 > >(point)[2] == GC_BGD) {
									fromSource = 0;
									toSink = 9999;
								}
								else {
									fromSource = -log(bgdGMM(color));
									toSink = -log(fgdGMM(color));
								}
								graph.addTermWeights(vtxIdx, fromSource, toSink);


								//ƽ����
								if (t > 0) {
									int pointN[6] = { t - 1,x,y,r,g,b };
									if (grid.at<Vec< int, 3 > >(pointN)[0] > 0) {
										double w = grid.at<Vec< int, 3 > >(point)[0] * grid.at<Vec< int, 3 > >(pointN)[0] + 1;
										w = 5 * log(w);
										int a = grid.at<Vec< int, 3 > >(pointN)[1];
										graph.addEdges(vtxIdx, grid.at<Vec< int, 3 > >(pointN)[1], w, w);
									}
								}
								if (x > 0) {
									int pointN[6] = { t,x - 1,y,r,g,b };
									if (grid.at<Vec< int, 3 >>(pointN)[0] > 0) {
										double w = grid.at<Vec< int, 3 > >(point)[0] * grid.at<Vec< int, 3 > >(pointN)[0] + 1;
										w = 5 * log(w);
										graph.addEdges(vtxIdx, grid.at<Vec< int, 3 > >(pointN)[1], w, w);
									}
								}
								if (y > 0) {
									int pointN[6] = { t,x,y - 1,r,g,b };
									if (grid.at<Vec< int, 3 > >(pointN)[0] > 0) {
										double w = grid.at<Vec< int, 3 > >(point)[0] * grid.at<Vec< int, 3 > >(pointN)[0] + 1;
										w = 5 * log(w);
										graph.addEdges(vtxIdx, grid.at<Vec< int, 3 > >(pointN)[1], w, w);
									}
								}
								if (r > 0) {
									int pointN[6] = { t,x,y,r - 1,g,b };
									if (grid.at<Vec< int, 3 > >(pointN)[0] > 0) {
										double w = grid.at<Vec< int, 3 > >(point)[0] * grid.at<Vec< int, 3 > >(pointN)[0] + 1;
										w = 2 * log(w);
										graph.addEdges(vtxIdx, grid.at<Vec< int, 3 > >(pointN)[1], w, w);
									}
								}
								if (g > 0) {
									int pointN[6] = { t,x,y,r,g - 1,b };
									if (grid.at<Vec< int, 3 > >(pointN)[0] > 0) {
										double w = grid.at<Vec< int, 3 > >(point)[0] * grid.at<Vec< int, 3 > >(pointN)[0] + 1;
										w = 2 * log(w);
										graph.addEdges(vtxIdx, grid.at<Vec< int, 3 > >(pointN)[1], w, w);
									}
								}
								if (b > 0) {
									int pointN[6] = { t,x,y,r,g,b - 1 };
									if (grid.at<Vec< int, 3 > >(pointN)[0] > 0) {
										double w = grid.at<Vec< int, 3 > >(point)[0] * grid.at<Vec< int, 3 > >(pointN)[0] + 1;
										w = 2 * log(w);
										graph.addEdges(vtxIdx, grid.at<Vec< int, 3 > >(pointN)[1], w, w);
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


int Bilateral::calculateVtxCount() {
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
							if (grid.at<Vec< int, 3 > >(point)[0] > 0) {
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


void Bilateral::estimateSegmentation(GCGraph<double>& graph, std::vector<Mat>& maskArr) {
	double _time = static_cast<double>(getTickCount());
	graph.maxFlow();//�����ͼ��
	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("ͼ��ָ���ʱ %f\n", _time);//��ʾʱ��

	double _time2 = static_cast<double>(getTickCount());
	
	_time2 = (static_cast<double>(getTickCount()) - _time2) / getTickFrequency();
	printf("grid�������mask��ʱ %f\n", _time2);//��ʾʱ��
}

void Bilateral::getGridPoint(int index, const Point p, int *point, int tSize, int xSize, int ySize) {
	point[0] = gridSize[0] * index / tSize;
	point[1] = gridSize[1] * p.x / xSize;
	point[2] = gridSize[2] * p.y / ySize;
	Vec3b color = (Vec3b)imgSrcArr[index].at<Vec3b>(p.x, p.y);
	point[3] = gridSize[3] * color[0] / 256;
	point[4] = gridSize[4] * color[1] / 256;
	point[5] = gridSize[5] * color[2] / 256;
}

void Bilateral::getGridPoint(int index, const Point p, std::vector<int>& point, int tSize, int xSize, int ySize) {
	point[0] = gridSize[0] * index / tSize;
	point[1] = gridSize[1] * p.y / xSize;
	point[2] = gridSize[2] * p.x / ySize;//x,y����������p���������µ����⡣
	Vec3b color = (Vec3b)imgSrcArr[index].at<Vec3b>(p);
	point[3] = gridSize[3] * color[0] / 256;
	point[4] = gridSize[4] * color[1] / 256;
	point[5] = gridSize[5] * color[2] / 256;
}

void Bilateral::run(std::vector<Mat>& maskArr) {
	for (int i = 0; i < imgSrcArr.size(); i++)
	{
		Mat mask = Mat::zeros(imgSrcArr[0].rows, imgSrcArr[0].cols, CV_8UC1);
		maskArr.push_back(mask);
	}

		GMM bgdGMM(bgModel), fgdGMM(fgModel);//ǰ����ģ��
		GCGraph<double> graph;//ͼ��
		constructGCGraph(bgdGMM, fgdGMM, graph);
		estimateSegmentation(graph, maskArr);


}