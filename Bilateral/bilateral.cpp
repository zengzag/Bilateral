#include "Bilateral.h"
#include <iostream>
#include <fstream>

Bilateral::Bilateral(std::vector<Mat> imgSrcArr, std::vector<Mat> maskArr) :
	imgSrcArr(imgSrcArr), maskArr(maskArr) {
	initGrid();
}

Bilateral::~Bilateral()
{
	for (int i = imgSrcArr.size() - 1;i >= 0;i--) {
		imgSrcArr[i].release();
	}
	imgSrcArr.clear();
	for (int i = maskArr.size() - 1;i >= 0;i--) {
		maskArr[i].release();
	}
	maskArr.clear();
	bgModelArr.clear();
	fgModelArr.clear();
	grid.release();
}

void Bilateral::InitGmms()
{
	double _time = static_cast<double>(getTickCount());//��ʱ

	for (int gmmNum = 0; gmmNum < gridSize[0]; gmmNum++) {//gmmNum��ģ�ͣ���̬ģ�ͣ�
		std::vector<Vec3f> bgdSamples;    //�ӱ�����洢������ɫ
		std::vector<Vec3f> fgdSamples;    //��ǰ����洢ǰ����ɫ
		std::vector<double> bgdWeight;    //�ӱ�����洢����Ȩֵ
		std::vector<double> fgdWeight;    //��ǰ����洢ǰ��Ȩֵ
		Mat bgModel,fgModel;
		bgModelArr.push_back(bgModel);
		fgModelArr.push_back(fgModel);

		for (int t = 0; t < gridSize[0]; t++) {
			for (int x = 0; x < gridSize[1]; x++) {
				for (int y = 0; y < gridSize[2]; y++) {
					for (int r = 0; r < gridSize[3]; r++) {
						for (int g = 0; g < gridSize[4]; g++) {
							for (int b = 0; b < gridSize[5]; b++) {			
								int point[6] = { t,x,y,r,g,b };
								int count = grid.at<Vec< int, 4 > >(point)[pixSum];
								if (count > 0) {
									Vec3f color = gridColor.at<Vec3f>(point);
									int vtxLable = grid.at<Vec< int, 4 > >(point)[lables];
									if (vtxLable == GC_BGD || vtxLable == GC_PR_BGD) {
										bgdSamples.push_back(color);
										double weight = vtxLable == GC_BGD ? 5.0 : 1.0;
										weight = weight*count*exp(-(t-gmmNum)*(t - gmmNum)) / 1000.0;
										//Ȩֵ��ȷ����ǰ������5������������Խ��ȨֵԽ��t���Ӧ��ǰģ��Խ��Խ�󣻳�100��ֹȨֵ���������
										bgdWeight.push_back(weight);
									}
									else {
										fgdSamples.push_back(color);
										double weight = vtxLable == GC_FGD ? 5.0 : 1.0;
										weight = weight*count*exp(-(t - gmmNum)*(t - gmmNum)) / 1000.0;
										fgdWeight.push_back(weight);
									}
								}
							}
						}
					}
				}
			}
		}

		//��˹ģ�ͽ���
		GMM bgdGMM(bgModelArr[gmmNum]), fgdGMM(fgModelArr[gmmNum]);
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

		//ѵ��GMMsģ��
		//for (int i = 0; i < (int)bgdSamples.size(); i++) {
		//	Vec3d color = bgdSamples[i];
		//	bgdLabels.at<int>(i, 0) = bgdGMM.whichComponent(color);
		//}

		//for (int i = 0; i < (int)fgdSamples.size(); i++) {
		//	Vec3d color = fgdSamples[i];
		//	fgdLabels.at<int>(i, 0) = fgdGMM.whichComponent(color);
		//}

		//bgdGMM.initLearning();
		//for (int i = 0; i < (int)bgdSamples.size(); i++)
		//	bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i], bgdWeight[i]);
		//bgdGMM.endLearning();
		//fgdGMM.initLearning();
		//for (int i = 0; i < (int)fgdSamples.size(); i++)
		//	fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i],fgdWeight[i]);
		//fgdGMM.endLearning();

	}

	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("��˹mģ�ͳ�ʼ����ʱ%f\n", _time);//��ʾʱ��
}

void Bilateral::nextGMMs() {
	double _time = static_cast<double>(getTickCount());//��ʱ
	for (int gmmNum = 0; gmmNum < gridSize[0]; gmmNum++) {//gmmNum��ģ�ͣ���̬ģ�ͣ�

		std::vector<Vec3f> bgdSamples;    //�ӱ�����洢������ɫ
		std::vector<Vec3f> fgdSamples;    //��ǰ����洢ǰ����ɫ
		std::vector<double> bgdWeight;    //�ӱ�����洢����Ȩֵ
		std::vector<double> fgdWeight;    //��ǰ����洢ǰ��Ȩֵ
		GMM bgdGMM(bgModelArr[gmmNum]), fgdGMM(fgModelArr[gmmNum]);
		std::vector<int> bgdLabels, fgdLabels; //��¼������ǰ����������������ÿ�����ض�ӦGMM���ĸ���˹ģ��

		for (int t = 0; t < gridSize[0]; t++) {
			for (int x = 0; x < gridSize[1]; x++) {
				for (int y = 0; y < gridSize[2]; y++) {
					for (int r = 0; r < gridSize[3]; r++) {
						for (int g = 0; g < gridSize[4]; g++) {
							for (int b = 0; b < gridSize[5]; b++) {
								int point[6] = { t,x,y,r,g,b };
								int count = grid.at<Vec< int, 4 > >(point)[pixSum];
								if (count > 0) {
									Vec3f color = gridColor.at<Vec3f>(point);
									int vtxLable = grid.at<Vec< int, 4 > >(point)[lables];
									if (vtxLable == GC_BGD || vtxLable == GC_PR_BGD) {
										bgdSamples.push_back(color);
										double weight = vtxLable == GC_BGD ? 5.0 : 1.0;
										weight = weight*count*exp(-(t - gmmNum)*(t - gmmNum)) / 1000.0;
										//Ȩֵ��ȷ����ǰ������5������������Խ��ȨֵԽ��t���Ӧ��ǰģ��Խ��Խ�󣻳�100��ֹȨֵ���������
										bgdWeight.push_back(weight);
										int label = bgdGMM.whichComponent(color);
										bgdLabels.push_back(label);
									}
									else {
										fgdSamples.push_back(color);
										double weight = vtxLable == GC_FGD ? 5.0 : 1.0;
										weight = weight*count*exp(-(t - gmmNum)*(t - gmmNum)) / 1000.0;
										fgdWeight.push_back(weight);
										int label = fgdGMM.whichComponent(color);
										fgdLabels.push_back(label);
									}
								}
							}
						}
					}
				}
			}
		}


		bgdGMM.initLearning();
		for (int i = 0; i < (int)bgdSamples.size(); i++)
			bgdGMM.addSample(bgdLabels[i], bgdSamples[i], bgdWeight[i]);
		bgdGMM.endLearning();
		fgdGMM.initLearning();
		for (int i = 0; i < (int)fgdSamples.size(); i++)
			fgdGMM.addSample(fgdLabels[i], fgdSamples[i], fgdWeight[i]);
		fgdGMM.endLearning();

	}

	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("��˹ģ���Ż���ʱ%f\n", _time);//��ʾʱ��
}


void Bilateral::initGrid() {
	double _time = static_cast<double>(getTickCount());

	Mat L(6, gridSize, CV_32SC(4), Scalar(0, -1, 0, -1));
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

				Vec3f colorMeans = gridColor.at<Vec3f>(point);//ƽ��ֵ��ⶥ����ɫ
				colorMeans[0] = colorMeans[0] * (count - 1.0) / (count + 0.0) + color[0] / (count + 0.0);
				colorMeans[1] = colorMeans[1] * (count - 1.0) / (count + 0.0) + color[1] / (count + 0.0);
				colorMeans[2] = colorMeans[2] * (count - 1.0) / (count + 0.0) + color[2] / (count + 0.0);
				gridColor.at<Vec3f>(point) = colorMeans;

				//ȷ������ı�ǩ��ǰ��������������ǰ�������ܱ�����
				int pixLable = maskArr[t].at<uchar>(x, y);
				int vtxLable = grid.at<Vec< int, 4 > >(point)[lables];
				if (pixLable != vtxLable && grid.at<Vec< int, 4 > >(point)[bothB_F] == 0) {
					if (vtxLable < 0) {
						grid.at<Vec< int, 4 > >(point)[lables] = pixLable;
					}
					else if ((pixLable == GC_BGD&&vtxLable == GC_FGD) || (pixLable == GC_FGD&&vtxLable == GC_BGD)) {
						grid.at<Vec< int, 4 > >(point)[bothB_F] = 1;
						grid.at<Vec< int, 4 > >(point)[lables] = GC_PR_FGD;
					}
					else if (pixLable < vtxLable) {
						grid.at<Vec< int, 4 > >(point)[lables] = pixLable;
					}
				}
			}
		}
	}

	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("����grid��ʱ%f\n", _time);//��ʾʱ��
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

void Bilateral::constructGCGraph(GCGraph<double>& graph) {
	double _time = static_cast<double>(getTickCount());

	double bata = calcBeta(imgSrcArr[0]);
	int vtxCount = calculateVtxCount();  //��������ÿһ��������һ������  
	int edgeCount = 2 * 64 * vtxCount;  //��������Ҫ����ͼ�߽�ıߵ�ȱʧ
	graph.create(vtxCount, edgeCount);

	for (int t = 0; t < gridSize[0]; t++) {

		GMM bgdGMM(bgModelArr[t]), fgdGMM(fgModelArr[t]);//ǰ����ģ��


		for (int x = 0; x < gridSize[1]; x++) {
			for (int y = 0; y < gridSize[2]; y++) {
				for (int r = 0; r < gridSize[3]; r++) {
					for (int g = 0; g < gridSize[4]; g++) {
						for (int b = 0; b < gridSize[5]; b++) {
							int point[6] = { t,x,y,r,g,b };

							if (grid.at<Vec< int, 4 > >(point)[pixSum] > 0) {
								int vtxIdx = graph.addVtx();//�������ص�ӳ��ͼӶ���

								//������
								grid.at<Vec< int, 4 > >(point)[vIdx] = vtxIdx;

								Vec3f color = gridColor.at<Vec3f>(point);
								double fromSource, toSink;

								int vtxLable = grid.at<Vec< int, 4 > >(point)[lables];
								//�ۺϷ���
								if (vtxLable  == GC_BGD) {
									fromSource = 0;
									toSink = 9999;
								}
								else if (vtxLable == GC_FGD) {
									fromSource = 9999;
									toSink = 0;
								}
								else {
									fromSource = -log(bgdGMM(color));
									toSink = -log(fgdGMM(color));
								}

								graph.addTermWeights(vtxIdx, fromSource, toSink);
														
								int count = 0;
								for (int tN = t; tN > t - 2 && tN >= 0; tN--) {
									for (int xN = x; xN > x - 2 && xN >= 0; xN--) {
										for (int yN = y; yN > y - 2 && yN >= 0; yN--) {
											for (int rN = r; rN > r - 2 && rN >= 0; rN--) {
												for (int gN = g; gN > g - 2 && gN >= 0; gN--) {
													for (int bN = b; bN > b - 2 && bN >= 0; bN--) {
														int pointN[6] = { tN,xN,yN,rN,gN,bN };
														count++;
														if (grid.at<Vec< int, 4 > >(pointN)[pixSum] > 0 && count > 1) {
															double num = grid.at<Vec< int, 4 > >(point)[pixSum] * grid.at<Vec< int, 4 > >(pointN)[pixSum] + 1;
															Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
															double e = exp(-bata*diff.dot(diff));  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�
															double w = 1.0 * e * log(num);
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
							if (grid.at<Vec< int, 4 > >(point)[0] > 0) {
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


void Bilateral::estimateSegmentation(GCGraph<double>& graph) {
	double _time = static_cast<double>(getTickCount());
	graph.maxFlow();//�����ͼ��
	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("ͼ��ָ���ʱ %f\n", _time);//��ʾʱ��

	double _time2 = static_cast<double>(getTickCount());

	for (int t = 0; t < gridSize[0]; t++) {
		for (int x = 0; x < gridSize[1]; x++) {
			for (int y = 0; y < gridSize[2]; y++) {
				for (int r = 0; r < gridSize[3]; r++) {
					for (int g = 0; g < gridSize[4]; g++) {
						for (int b = 0; b < gridSize[5]; b++) {
							int point[6] = { t,x,y,r,g,b };
							if (grid.at<Vec< int, 4 > >(point)[pixSum] > 0) {
								int vertex = grid.at<Vec< int, 4 > >(point)[vIdx];
								if (graph.inSourceSegment(vertex))
									grid.at<Vec< int, 4 > >(point)[lables] = grid.at<Vec< int, 4 > >(point)[lables] == GC_FGD ? GC_FGD : GC_PR_FGD;
								else
									grid.at<Vec< int, 4 > >(point)[lables] = grid.at<Vec< int, 4 > >(point)[lables] == GC_BGD ? GC_BGD : GC_PR_BGD;
							}
						}
					}
				}
			}
		}
	}	

	_time2 = (static_cast<double>(getTickCount()) - _time2) / getTickFrequency();
	printf("���grid�ָ�����ʱ %f\n", _time2);//��ʾʱ��
}

void Bilateral::getMask() {
	double _time = static_cast<double>(getTickCount());

	int tSize = maskArr.size();
	int xSize = imgSrcArr[0].rows;
	int ySize = imgSrcArr[0].cols;
	for (int t = 0; t < tSize; t++)
	{
		for (int y = 0; y < ySize; y++)
		{
#pragma omp parallel for
			for (int x = 0; x < xSize; x++)
			{
				Point p(x, y);
				int point[6] = { 0,0,0,0,0,0 };
				getGridPoint(t, p, point, tSize, xSize, ySize);
				maskArr[t].at<uchar>(p.x, p.y) = grid.at<Vec< int, 4 > >(point)[lables] & 1;//�����㣬ȡ���λ
			}
		}
	}

	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("grid�������mask��ʱ %f\n", _time);//��ʾʱ��
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


void Bilateral::run(int times) {
	InitGmms();
	for (int i = 0; i < times; i++)
	{
		std::cout << "��" <<i+1<<"�ε���" << std::endl;
		nextGMMs();
		GCGraph<double> graph;
		constructGCGraph(graph);
		estimateSegmentation(graph);
		
	}	
	//GCGraph<double> graph;
	//constructGCGraph(graph);
	//estimateSegmentation(graph);
	getMask();
}