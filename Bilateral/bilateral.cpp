#include "Bilateral.h"
#include <iostream>
#include <fstream>

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
	bgModelArr.clear();
	fgModelArr.clear();
	grid.release();
}

void Bilateral::InitGmms(std::vector<Mat>& maskArr, int* index)
{
	double _time = static_cast<double>(getTickCount());//计时

	int gmmSize = maskArr.size();
	int tSize = imgSrcArr.size();
	int xSize = imgSrcArr[0].rows;
	int ySize = imgSrcArr[0].cols;
	int point[6] = { 0,0,0,0,0,0 };

	for (int t = 0; t < gmmSize; t++) {
		for (int x = 0; x < xSize; x++)
		{
			for (int y = 0; y < ySize; y++)
			{
				uchar a = maskArr[t].at<uchar>(x, y);
				if (maskArr[t].at<uchar>(x, y) == GC_BGD) {
					getGridPoint(index[t], Point(x, y), point, tSize, xSize, ySize);
					grid.at<Vec< int, 4 > >(point)[bgdSum] += 3;
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
						int pointN[6] = { point[0],point[1],point[2] ,point[3],point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[3] < gridSize[3] - 1) {
						int pointN[6] = { point[0] ,point[1],point[2],point[3] + 1,point[4],point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[4] < gridSize[4] - 1) {
						int pointN[6] = { point[0],point[1] + 1,point[2],point[3],point[4] + 1,point[5] };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
					if (point[5] < gridSize[5] - 1) {
						int pointN[6] = { point[0],point[1],point[2],point[3],point[4],point[5] + 1 };
						grid.at<Vec< int, 4 > >(pointN)[bgdSum] += 1;
					}
				}
				else if (maskArr[t].at<uchar>(x, y) == GC_FGD/*GC_FGD*/) {
					getGridPoint(index[t], Point(x, y), point, tSize, xSize, ySize);
					grid.at<Vec< int, 4 > >(point)[fgdSum] += 3;
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
			}
		}
	}

	std::vector<Vec3f> bgdSamples;    //从背景点存储背景颜色
	std::vector<Vec3f> fgdSamples;    //从前景点存储前景颜色
	std::vector<std::vector<double> > bgdWeight(gmmSize);    //从背景点存储背景权值
	std::vector<std::vector<double> > fgdWeight(gmmSize);    //从前景点存储前景权值

	for (int t = 0; t < gridSize[0]; t++) {
		for (int x = 0; x < gridSize[1]; x++) {
			for (int y = 0; y < gridSize[2]; y++) {
				for (int r = 0; r < gridSize[3]; r++) {
					for (int g = 0; g < gridSize[4]; g++) {
						for (int b = 0; b < gridSize[5]; b++) {
							int point[6] = { t,x,y,r,g,b };
							int bgdcount = grid.at<Vec< int, 4 > >(point)[bgdSum];
							int fgdcount = grid.at<Vec< int, 4 > >(point)[fgdSum];
							if (bgdcount > 0) {
								Vec3f color = gridColor.at<Vec3f>(point);
								bgdSamples.push_back(color);
								for (int tGmm = 0; tGmm < gmmSize; tGmm++) {
									double weight = bgdcount*exp(-2 * (t / 3 - tGmm)*(t / 3 - tGmm));
									//权值：确定的前背景乘5；包含的像素越多权值越大；t与对应当前模型越近越大；除100防止权值过大溢出。
									bgdWeight[tGmm].push_back(weight);
								}
							}
							if (fgdcount > 0) {
								Vec3f color = gridColor.at<Vec3f>(point);
								fgdSamples.push_back(color);
								for (int tGmm = 0; tGmm < gmmSize; tGmm++) {
									double weight = fgdcount*exp(-2 * (t/3 - tGmm)*(t/3 - tGmm));
									//权值：确定的前背景乘5；包含的像素越多权值越大；t与对应当前模型越近越大；除100防止权值过大溢出。
									fgdWeight[tGmm].push_back(weight);
								}
							}
						}
					}
				}
			}
		}
	}

	for (int tGmm = 0; tGmm < gmmSize; tGmm++) {
		Mat bgModel, fgModel;
		GMM bgdGMM(bgModel), fgdGMM(fgModel);

		const int kMeansItCount = 10;  //迭代次数  
		const int kMeansType = KMEANS_PP_CENTERS; //Use kmeans++ center initialization by Arthur and Vassilvitskii  
		Mat bgdLabels, fgdLabels; //记录背景和前景的像素样本集中每个像素对应GMM的哪个高斯模型

								  //kmeans进行分类
		Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
		kmeans(_bgdSamples, GMM::componentsCount, bgdLabels,
			TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);
		Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
		kmeans(_fgdSamples, GMM::componentsCount, fgdLabels,
			TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);

		//分类后的结果用来训练GMMs（初始化）
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
			//训练GMMs模型
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

	}
	
	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("高斯建模用时%f\n", _time);//显示时间
}

void Bilateral::initGrid() {
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
	printf("构建grid用时%f\n", _time);//显示时间
}

static double calcBeta(const Mat& img)
{
	double beta = 0;
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			//计算四个方向邻域两像素的差别，也就是欧式距离或者说二阶范数  
			//（当所有像素都算完后，就相当于计算八邻域的像素差了）  
			Vec3d color = img.at<Vec3b>(y, x);
			if (x > 0) // left  >0的判断是为了避免在图像边界的时候还计算，导致越界  
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x - 1);
				beta += diff.dot(diff);  //矩阵的点乘，也就是各个元素平方的和  
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
		beta = 1.f / (2 * beta / (4 * img.cols*img.rows - 3 * img.cols - 3 * img.rows + 2)); //论文公式（5）  

	return beta;
}

void Bilateral::constructGCGraph(GCGraph<double>& graph) {
	double _time = static_cast<double>(getTickCount());

	double bata = calcBeta(imgSrcArr[0]);
	int vtxCount = calculateVtxCount();  //顶点数，每一个像素是一个顶点  
	int edgeCount = 2 * 64 * vtxCount;  //边数，需要考虑图边界的边的缺失
	graph.create(vtxCount, edgeCount);
	int eCount = 0;
	for (int t = 0; t < gridSize[0]; t++) {
		int gmmT = t / 3;
		GMM bgdGMM(bgModelArr[gmmT]), fgdGMM(fgModelArr[gmmT]);

		for (int x = 0; x < gridSize[1]; x++) {
			for (int y = 0; y < gridSize[2]; y++) {
				for (int r = 0; r < gridSize[3]; r++) {
					for (int g = 0; g < gridSize[4]; g++) {
						for (int b = 0; b < gridSize[5]; b++) {
							int point[6] = { t,x,y,r,g,b };
							int pixCount = grid.at<Vec< int, 4 > >(point)[pixSum];
							if (pixCount > 0) {
								int vtxIdx = graph.addVtx();//存在像素点映射就加顶点

								//先验项
								grid.at<Vec< int, 4 > >(point)[vIdx] = vtxIdx;

								Vec3f color = gridColor.at<Vec3f>(point);
								double fromSource, toSink;

								double toSinkSum = grid.at<Vec< int, 4 > >(point)[fgdSum];
								double fromSourceSum = grid.at<Vec< int, 4 > >(point)[bgdSum];
								//综合方法
								if (fromSourceSum > 20 && toSinkSum == 0) {
									fromSource = 0;
									toSink = 9999;
								}
								else if (fromSourceSum == 0 && toSinkSum > 20) {
									fromSource = 9999;
									toSink = 0;
								}
								else {
									fromSource = -log(bgdGMM(color))*sqrt(pixCount);
									toSink = -log(fgdGMM(color))*sqrt(pixCount);
								}

								//原方法
								/*fromSource =  (toSinkSum + 2);
								toSink =  (fromSourceSum + 2);*/


								graph.addTermWeights(vtxIdx, fromSource, toSink);


								//平滑项
								for (int tN = t; tN > t - 2 && tN >= 0&& tN < gridSize[0]; tN--) {
									for (int xN = x ; xN > x - 2 && xN >= 0 && xN < gridSize[1]; xN--) {
										for (int yN = y ; yN > y - 2 && yN >= 0 && yN < gridSize[2]; yN--) {
											for (int rN = r; rN > r - 2 && rN >= 0 && rN < gridSize[3]; rN--) {
												for (int gN = g; gN > g - 2 && gN >= 0 && gN < gridSize[4]; gN--) {
													for (int bN = b ; bN > b - 2 && bN >= 0 && bN < gridSize[5]; bN--) {
														int pointN[6] = { tN,xN,yN,rN,gN,bN };
														int vtxIdxNew = grid.at<Vec< int, 4 > >(pointN)[vIdx];
														if (grid.at<Vec< int, 4 > >(pointN)[pixSum]>0 && vtxIdxNew > 0 && vtxIdxNew != vtxIdx) {
															double num = sqrt(grid.at<Vec< int, 4 > >(point)[pixSum] * grid.at<Vec< int, 4 > >(pointN)[pixSum] + 1);
															Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
															double e = exp(-bata*diff.dot(diff));  //矩阵的点乘，也就是各个元素平方的和
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

							}
						}
					}
				}

			}
		}
	}

	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("图割构图用时 %f\n", _time);//显示时间
	printf("顶点的总数 %d\n", vtxCount);
	printf("边的总数 %d\n", eCount);
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
	graph.maxFlow();//最大流图割
	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("图割分割用时 %f\n", _time);//显示时间

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
	printf("grid结果传递mask用时 %f\n", _time2);//显示时间
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
	point[2] = gridSize[2] * p.x / ySize;//x,y互换、由于p坐标存错，导致的问题。
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
				Vec3b color;//计算grid中顶点对应的颜色
				color[0] = (r * 256 + 256 / 2) / gridSize[3];//多加256/2是为了把颜色移到方格中心
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




void Bilateral::run(std::vector<Mat>& maskArr) {
	for (int i = 0; i < imgSrcArr.size(); i++)
	{
		Mat mask = Mat::zeros(imgSrcArr[0].rows, imgSrcArr[0].cols, CV_8UC1);
		maskArr.push_back(mask);
	}
	GCGraph<double> graph;//图割

	constructGCGraph(graph);
	//getColor();
	estimateSegmentation(graph, maskArr);


}