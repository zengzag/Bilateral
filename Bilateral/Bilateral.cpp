#include "Bilateral.h"
#include <iostream>
#include <fstream>

Bilateral::Bilateral(std::vector<Mat> img) :
	imgSrcArr(img) {
	initGrid();
}

Bilateral::~Bilateral()
{
	imgSrcArr.clear();
	bgModelArr.clear();
	fgModelArr.clear();
	grid.release();
}

void Bilateral::InitGmms(std::vector<Mat>& maskArr, int* index)
{
	double _time = static_cast<double>(getTickCount());//��ʱ

	int maskSize = maskArr.size();
	int tSize = imgSrcArr.size();
	int xSize = imgSrcArr[0].rows;
	int ySize = imgSrcArr[0].cols;
	int point[6] = { 0,0,0,0,0,0 };

	for (int t = 0; t < maskSize; t++) {
		for (int x = 0; x < xSize; x++)
		{
			for (int y = 0; y < ySize; y++)
			{
				uchar a = maskArr[t].at<uchar>(x, y);
				if (maskArr[t].at<uchar>(x, y) == GC_BGD) {
					getGridPoint(index[t], Point(x, y), point, tSize, xSize, ySize);
					grid.at<Vec< int, 4 > >(point)[bgdSum] += 3;
				}
				else if (maskArr[t].at<uchar>(x, y) == GC_FGD/*GC_FGD*/) {
					getGridPoint(index[t], Point(x, y), point, tSize, xSize, ySize);
					grid.at<Vec< int, 4 > >(point)[fgdSum] += 3;
				}
				else if (maskArr[t].at<uchar>(x, y) == GC_PR_FGD/*GC_FGD*/) {
					getGridPoint(index[t], Point(x, y), point, tSize, xSize, ySize);
					grid.at<Vec< int, 4 > >(point)[fgdSum] += 1;
				}
				else if (maskArr[t].at<uchar>(x, y) == GC_PR_BGD/*GC_FGD*/) {
					getGridPoint(index[t], Point(x, y), point, tSize, xSize, ySize);
					grid.at<Vec< int, 4 > >(point)[bgdSum] += 1;
				}
			}
		}
	}

	std::vector<Vec3f> bgdSamples;    //�ӱ�����洢������ɫ
	std::vector<Vec3f> fgdSamples;    //��ǰ����洢ǰ����ɫ
	std::vector<std::vector<double> > bgdWeight(gridSize[0]);    //�ӱ�����洢����Ȩֵ
	std::vector<std::vector<double> > fgdWeight(gridSize[0]);    //��ǰ����洢ǰ��Ȩֵ

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
								if (bgdcount > (pixCount / 2)) {
									Vec3f color = gridColor.at<Vec3f>(point);
									bgdSamples.push_back(color);
									for (int tGmm = 0; tGmm < gridSize[0]; tGmm++) {
										double weight = pixCount*exp(-2 * (t - tGmm)*(t - tGmm))*(bgdcount / (fgdcount + bgdcount));
										bgdWeight[tGmm].push_back(weight);
									}
								}
								if (fgdcount > (pixCount / 2)) {
									Vec3f color = gridColor.at<Vec3f>(point);
									fgdSamples.push_back(color);
									for (int tGmm = 0; tGmm < gridSize[0]; tGmm++) {
										double weight = pixCount*exp(-2 * (t - tGmm)*(t - tGmm))*(fgdcount / (fgdcount + bgdcount));
										fgdWeight[tGmm].push_back(weight);
									}
								}
							}
						}
					}
				}
			}
		}
	}

	for (int tGmm = 0; tGmm < gridSize[0]; tGmm++) {
		Mat bgModel, fgModel;
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
			bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i], bgdWeight[tGmm][i]);
		bgdGMM.endLearning();
		fgdGMM.initLearning();
		for (int i = 0; i < (int)fgdSamples.size(); i++)
			fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i], fgdWeight[tGmm][i]);
		fgdGMM.endLearning();
		for (int times = 0; times < 5; times++)
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
				bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i], bgdWeight[tGmm][i]);
			bgdGMM.endLearning();
			fgdGMM.initLearning();
			for (int i = 0; i < (int)fgdSamples.size(); i++)
				fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i], fgdWeight[tGmm][i]);
			fgdGMM.endLearning();
		}

		bgModelArr.push_back(bgModel);
		fgModelArr.push_back(fgModel);


		std::vector<Vec3f> unSamples;    //��������
		std::vector<double>  unWeight;

		for (int i = 0; i < (int)bgdSamples.size(); i++) {
			Vec3d color = bgdSamples[i];
			double b = bgdGMM(color), f = fgdGMM(color);
			if (b < f) {
				unSamples.push_back(color);
				unWeight.push_back((f - b) / f);
			}
		}
		for (int i = 0; i < (int)fgdSamples.size(); i++) {
			Vec3d color = fgdSamples[i];
			double b = bgdGMM(color), f = fgdGMM(color);
			if (b > f) {
				unSamples.push_back(color);
				unWeight.push_back((b - f) / b);
			}
		}
		Mat unModel;
		GMM unGMM(unModel);
		Mat unLabels;
		Mat _unSamples((int)unSamples.size(), 3, CV_32FC1, &unSamples[0][0]);
		kmeans(_unSamples, GMM::componentsCount, unLabels,
			TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);
		unGMM.initLearning();
		for (int i = 0; i < (int)unSamples.size(); i++)
			unGMM.addSample(unLabels.at<int>(i, 0), unSamples[i], unWeight[i]);
		unGMM.endLearning();
		for (int times = 0; times < 3; times++)
		{
			//ѵ��GMMsģ��
			for (int i = 0; i < (int)unSamples.size(); i++) {
				Vec3d color = unSamples[i];
				unLabels.at<int>(i, 0) = unGMM.whichComponent(color);
			}
			unGMM.initLearning();
			for (int i = 0; i < (int)unSamples.size(); i++)
				unGMM.addSample(unLabels.at<int>(i, 0), unSamples[i], unWeight[i]);
			unGMM.endLearning();
		}
		unModelArr.push_back(unModel);

	}

	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("��˹��ģ��ʱ%f\n", _time);//��ʾʱ��
}

void Bilateral::initGrid() {
	double _time = static_cast<double>(getTickCount());

	Mat L(6, gridSize, CV_32SC(4), Scalar(0, 0, 0, -1));
	Mat C(6, gridSize, CV_32FC(3), Scalar::all(0));
	Mat P(6, gridSize, CV_32FC(3), Scalar::all(0));
	grid = L;gridColor = C;gridProbable = P;
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
	int edgeCount = 2 * 256 * vtxCount;  //��������Ҫ����ͼ�߽�ıߵ�ȱʧ
	graph.create(vtxCount, edgeCount);
	int eCount = 0, eCount2 = 0, eCount3 = 0;

	for (int t = 0; t < gridSize[0]; t++) {
		int gmmT = t;
		GMM bgdGMM(bgModelArr[gmmT]), fgdGMM(fgModelArr[gmmT]), unGMM(unModelArr[gmmT]);
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
								double unWeight; // ��ɫģ��Ȩ�ء�
								//�ۺϷ���
								if ((bSum > pixCount) && fSum == 0) {
									fromSource = 0;
									toSink = 9999;
								}
								else if (bSum == 0 && (fSum > pixCount)) {
									fromSource = 9999;
									toSink = 0;
								}
								else {
									double bgd = bgdGMM(color);
									double fgd = fgdGMM(color);
									double un = unGMM(color);//��ɫģ�͵Ŀ��Ŷ�,Խ��Խ�����š�
									unWeight = 1.0 - (un / (bgd + fgd + un));//��ɫģ��Ȩ�ء�
									double sumWeight = abs(bSum - fSum) / (bSum + fSum + 1.0);//���Ȩ��
									if (unWeight < 0.3) {
										bgd = fgd;
										eCount3++;
									}
									//unWeight = 0.5;	sumWeight = 0.5;
									gridProbable.at<Vec3f>(point)[0] = fgd / (bgd + fgd);
									gridProbable.at<Vec3f>(point)[1] = (fSum + 1.0) / (fSum + bSum + 2.0);//���ʿ��ӻ�

									fromSource = (-log(bgd / (bgd + fgd))*unWeight - log((bSum + 1.0) / (fSum + bSum + 2.0))*sumWeight)*sqrt(pixCount);
									toSink = (-log(fgd / (bgd + fgd))*unWeight - log((fSum + 1.0) / (fSum + bSum + 2.0))*sumWeight)*sqrt(pixCount);

									gridProbable.at<Vec3f>(point)[2] = (fromSource) / (fromSource + toSink);
								}
								graph.addTermWeights(vtxIdx, fromSource, toSink);

								//ƽ����
								for (int tN = t; tN > t - 2 && tN >= 0 && tN < gridSize[0]; tN--) {
									for (int xN = x; xN > x - 2 && xN >= 0 && xN < gridSize[1]; xN--) {
										for (int yN = y; yN > y - 2 && yN >= 0 && yN < gridSize[2]; yN--) {
											for (int rN = r; rN > r - 2 && rN >= 0 && rN < gridSize[3]; rN--) {
												for (int gN = g; gN > g - 2 && gN >= 0 && gN < gridSize[4]; gN--) {
													for (int bN = b; bN > b - 2 && bN >= 0 && bN < gridSize[5]; bN--) {
														int pointN[6] = { tN,xN,yN,rN,gN,bN };
														int vtxIdxNew = grid.at<Vec< int, 4 > >(pointN)[vIdx];
														if (grid.at<Vec< int, 4 > >(pointN)[pixSum] > 0 && vtxIdxNew > 0 && vtxIdxNew != vtxIdx) {
															double num = sqrt(grid.at<Vec< int, 4 > >(point)[pixSum] * grid.at<Vec< int, 4 > >(pointN)[pixSum] + 1);
															Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
															double e = exp(-bata*diff.dot(diff));  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�
															double w = 1.0 * e * sqrt(num);
															graph.addEdges(vtxIdx, vtxIdxNew, w, w);
															eCount++;
														}
													}
												}
											}
										}
									}
								}

								//for (int tN = t - 1;tN >= 0;tN--) {
								//	int pointN[6] = { tN,x,y,r,g,b };
								//	int vtxIdxNew = grid.at<Vec< int, 4 > >(pointN)[vIdx];
								//	if (grid.at<Vec< int, 4 > >(pointN)[pixSum] > 0 && vtxIdxNew > 0 && vtxIdxNew != vtxIdx) {
								//		double num = sqrt(grid.at<Vec< int, 4 > >(point)[pixSum] * grid.at<Vec< int, 4 > >(pointN)[pixSum] + 1);
								//		Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
								//		double e = exp(-bata*diff.dot(diff));  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�
								//		double w = 1.0 * e * sqrt(num);
								//		graph.addEdges(vtxIdx, vtxIdxNew, w, w);										
								//		eCount++;
								//		break;
								//	}
								//}
								//for (int xN = t - 1;xN >= 0;xN--) {
								//	int pointN[6] = { t,xN,y,r,g,b };
								//	int vtxIdxNew = grid.at<Vec< int, 4 > >(pointN)[vIdx];
								//	if (grid.at<Vec< int, 4 > >(pointN)[pixSum] > 0 && vtxIdxNew > 0 && vtxIdxNew != vtxIdx) {
								//		double num = sqrt(grid.at<Vec< int, 4 > >(point)[pixSum] * grid.at<Vec< int, 4 > >(pointN)[pixSum] + 1);
								//		Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
								//		double e = exp(-bata*diff.dot(diff));  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�
								//		double w = 1.0 * e * sqrt(num);
								//		graph.addEdges(vtxIdx, vtxIdxNew, w, w);
								//		eCount++;
								//		break;
								//	}
								//}
								//for (int yN = t - 1;yN >= 0;yN--) {
								//	int pointN[6] = { t,x,yN,r,g,b };
								//	int vtxIdxNew = grid.at<Vec< int, 4 > >(pointN)[vIdx];
								//	if (grid.at<Vec< int, 4 > >(pointN)[pixSum] > 0 && vtxIdxNew > 0 && vtxIdxNew != vtxIdx) {
								//		double num = sqrt(grid.at<Vec< int, 4 > >(point)[pixSum] * grid.at<Vec< int, 4 > >(pointN)[pixSum] + 1);
								//		Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
								//		double e = exp(-bata*diff.dot(diff));  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�
								//		double w = 1.0 * e * sqrt(num);
								//		graph.addEdges(vtxIdx, vtxIdxNew, w, w);
								//		eCount++;
								//		break;
								//	}
								//}
								//for (int rN = t - 1;rN >= 0;rN--) {
								//	int pointN[6] = { t,x,y,rN,g,b };
								//	int vtxIdxNew = grid.at<Vec< int, 4 > >(pointN)[vIdx];
								//	if (grid.at<Vec< int, 4 > >(pointN)[pixSum] > 0 && vtxIdxNew > 0 && vtxIdxNew != vtxIdx) {
								//		double num = sqrt(grid.at<Vec< int, 4 > >(point)[pixSum] * grid.at<Vec< int, 4 > >(pointN)[pixSum] + 1);
								//		Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
								//		double e = exp(-bata*diff.dot(diff));  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�
								//		double w = 10.0 * e * sqrt(num);
								//		graph.addEdges(vtxIdx, vtxIdxNew, w, w);
								//		eCount++;
								//		break;
								//	}
								//}
								//for (int gN = t - 1;gN >= 0;gN--) {
								//	int pointN[6] = { t,x,y,r,gN,b };
								//	int vtxIdxNew = grid.at<Vec< int, 4 > >(pointN)[vIdx];
								//	if (grid.at<Vec< int, 4 > >(pointN)[pixSum] > 0 && vtxIdxNew > 0 && vtxIdxNew != vtxIdx) {
								//		double num = sqrt(grid.at<Vec< int, 4 > >(point)[pixSum] * grid.at<Vec< int, 4 > >(pointN)[pixSum] + 1);
								//		Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
								//		double e = exp(-bata*diff.dot(diff));  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�
								//		double w = 10.0 * e * sqrt(num);
								//		graph.addEdges(vtxIdx, vtxIdxNew, w, w);
								//		eCount++;
								//		break;
								//	}
								//}
								//for (int bN = t - 1;bN >= 0;bN--) {
								//	int pointN[6] = { t,x,y,r,g,bN };
								//	int vtxIdxNew = grid.at<Vec< int, 4 > >(pointN)[vIdx];
								//	if (grid.at<Vec< int, 4 > >(pointN)[pixSum] > 0 && vtxIdxNew > 0 && vtxIdxNew != vtxIdx) {
								//		double num = sqrt(grid.at<Vec< int, 4 > >(point)[pixSum] * grid.at<Vec< int, 4 > >(pointN)[pixSum] + 1);
								//		Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
								//		double e = exp(-bata*diff.dot(diff));  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�
								//		double w = 10.0 * e * sqrt(num);
								//		graph.addEdges(vtxIdx, vtxIdxNew, w, w);
								//		eCount++;
								//		break;
								//	}
								//}

								/*if (unWeight>0) {
									for (int tN = t; tN >= 0 && tN > t - 2;tN--) {
										for (int xN = 0; xN < x; xN++) {
											for (int yN = 0; yN < gridSize[2]; yN++) {
												int pointN[6] = { tN,xN,yN,r,g,b };
												int vtxIdxNew = grid.at<Vec< int, 4 > >(pointN)[vIdx];
												int vNewPixCount = grid.at<Vec< int, 4 > >(pointN)[pixSum];

												if (vNewPixCount > 0 && vtxIdxNew > 0 && vtxIdxNew != vtxIdx) {
													double vPixSumDiff = (double)pixCount / (double)vNewPixCount;

													Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
													double colorDst = (diff.dot(diff));
													if (vPixSumDiff > 0.8 && vPixSumDiff < 1.25 && colorDst < 64.0) {
														double w = 0.4 * exp(-bata*colorDst) * sqrt(vNewPixCount);
														graph.addEdges(vtxIdx, vtxIdxNew, w, w);
														eCount++;
														eCount2++;
													};
												}
											}
										}
									}
								}*/

							}
						}
					}
				}

			}
		}
	}

	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("ͼ�ͼ��ʱ %f\n", _time);//��ʾʱ��
	printf("��������� %d\n", vtxCount);
	printf("�ߵ����� %d\n", eCount);
	printf("e3������ %d\n", eCount2);
	printf("unWeight<0.5������ %d\n", eCount3);
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


void Bilateral::estimateSegmentation(GCGraph<double>& graph, std::vector<Mat>& maskArr) {
	double _time = static_cast<double>(getTickCount());
	graph.maxFlow();//�����ͼ��
	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("ͼ��ָ���ʱ %f\n", _time);//��ʾʱ��

	double _time2 = static_cast<double>(getTickCount());
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
				int vertex = grid.at<Vec< int, 4 > >(point)[vIdx];
				if (graph.inSourceSegment(vertex))
					maskArr[t].at<uchar>(p.x, p.y) = 1;
				else
					maskArr[t].at<uchar>(p.x, p.y) = 0;
			}
		}
	}

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
	point[2] = gridSize[2] * p.x / ySize;//x,y����������p�����������µ����⡣
	Vec3b color = (Vec3b)imgSrcArr[index].at<Vec3b>(p);
	point[3] = gridSize[3] * color[0] / 256;
	point[4] = gridSize[4] * color[1] / 256;
	point[5] = gridSize[5] * color[2] / 256;
}


void Bilateral::getColor() {
	std::ofstream f1("E:/Projects/OpenCV/DAVIS-data/image/1color.txt");
	if (!f1)return;

	for (int r = 0; r < gridSize[3]; r++) {
		for (int g = 0; g < gridSize[4]; g++) {
			for (int b = 0; b < gridSize[5]; b++) {
				f1 << "---------------------------------" << std::endl;
				Vec3b color;//����grid�ж����Ӧ����ɫ
				color[0] = (r * 256 + 256 / 2) / gridSize[3];//���256/2��Ϊ�˰���ɫ�Ƶ���������
				color[1] = (g * 256 + 256 / 2) / gridSize[4];
				color[2] = (b * 256 + 256 / 2) / gridSize[5];
				f1 << (int)color[0] << "\t" << (int)color[1] << "\t" << (int)color[2] << std::endl;

				for (int t = 0; t < gridSize[0]; t++) {
					for (int x = 0; x < gridSize[1]; x++) {
						for (int y = 0; y < gridSize[2]; y++) {

							int point[6] = { t,x,y,r,g,b };
							if (grid.at<Vec< int, 4 > >(point)[pixSum] != -1) {
								Vec3f colorM = gridColor.at<Vec3f>(point);
								f1 << (float)colorM[0] << "\t" << (float)colorM[1] << "\t" << (float)colorM[2] << std::endl;

							}
						}
					}
				}
			}
		}
	}
	f1.close();
}

void Bilateral::getGmmProMask(std::vector<Mat>& maskArr) {
	for (int i = 0; i < imgSrcArr.size(); i++)
	{
		Mat mask = Mat::zeros(imgSrcArr[0].rows, imgSrcArr[0].cols, CV_8UC1);
		maskArr.push_back(mask);
	}
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
				float probable = gridProbable.at<Vec3f>(point)[0];
				maskArr[t].at<uchar>(p.x, p.y) = (uchar)(probable * 255);
			}
		}
	}
}

void Bilateral::getKeyProMask(std::vector<Mat>& maskArr) {
	for (int i = 0; i < imgSrcArr.size(); i++)
	{
		Mat mask = Mat::zeros(imgSrcArr[0].rows, imgSrcArr[0].cols, CV_8UC1);
		maskArr.push_back(mask);
	}
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
				float probable = gridProbable.at<Vec3f>(point)[1];
				maskArr[t].at<uchar>(p.x, p.y) = (uchar)(probable * 255);
			}
		}
	}
}

void Bilateral::getTotalProMask(std::vector<Mat>& maskArr) {
	for (int i = 0; i < imgSrcArr.size(); i++)
	{
		Mat mask = Mat::zeros(imgSrcArr[0].rows, imgSrcArr[0].cols, CV_8UC1);
		maskArr.push_back(mask);
	}
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
				float probable = gridProbable.at<Vec3f>(point)[2];
				maskArr[t].at<uchar>(p.x, p.y) = (uchar)(probable * 255);
			}
		}
	}
}


void Bilateral::run(std::vector<Mat>& maskArr) {
	for (int i = 0; i < imgSrcArr.size(); i++)
	{
		Mat mask = Mat::zeros(imgSrcArr[0].rows, imgSrcArr[0].cols, CV_8UC1);
		maskArr.push_back(mask);
	}
	GCGraph<double> graph;//ͼ��

	constructGCGraph(graph);
	//getColor();
	estimateSegmentation(graph, maskArr);

}