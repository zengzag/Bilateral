#pragma once
#include "opencv2/imgproc.hpp"
#include "GMM.h"
#include "gcgraph.hpp"

#ifndef _BILATERAL_H_
#define _BILATERAL_H_

using namespace cv;

class Bilateral
{
public:
	std::vector<Mat> imgSrcArr;	 //输入图片数据
	Mat bgModel, fgModel;	//前背景高斯模型
	Mat keyMask; //关键帧的mask
	Mat grid;	//升维，平均取点，得到的grid。6维数组，保存顶点值与邻近像素点总数。
	const int gridSize[6] = { 3,20,30,24,24,24 };	//grid各个维度的大小,按顺序来为：t,x,y,r,g,b。
public:
	Bilateral(std::vector<Mat> img);
	~Bilateral();
	void InitGmms(Mat& , int);
	void run(std::vector<Mat>& );
private:
	void initGrid();
	void nextGMMs();
	void constructGCGraph(const GMM&, const GMM&, GCGraph<double>& graph);
	int calculateVtxCount();
	void estimateSegmentation(GCGraph<double>&, std::vector<Mat>& );
	void getGridPoint(int , const Point , int *, int , int , int );
	void getGridPoint(int , const Point , std::vector<int>& , int , int , int );
};

#endif