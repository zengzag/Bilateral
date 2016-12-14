#pragma once
#include "opencv2/imgproc.hpp"
#include "GMM.h"
#include "gcgraph.hpp"
#include "Bilateral.h"

class BilateralSimple
{
public:
	std::vector<Mat> imgSrcArr;	 //输入图片数据
	Mat bgModel, fgModel, unModel;	//前背景高斯模型
	Mat grid, gridColor;	//升维，平均取点，得到的grid。6维数组，保存顶点值与邻近像素点总数。
	const int gridSize[6] = { 1,30,50,16,16,16 };	//grid各个维度的大小,按顺序来为：t,x,y,r,g,b。

public:
	BilateralSimple(std::vector<Mat> img);
	~BilateralSimple();
	void InitGmms(Mat& maskArr);
	void run(Mat&);
private:
	void initGrid();
	void constructGCGraph(GCGraph<double>& graph);
	int calculateVtxCount();
	void estimateSegmentation(GCGraph<double>&, Mat&);
	void getGridPoint(int, const Point, int *, int, int, int);
	void getGridPoint(int, const Point, std::vector<int>&, int, int, int);
};

