#pragma once
#include "opencv2/imgproc.hpp"
#include "GMM.h"
#include "gcgraph.hpp"

#ifndef _BILATERAL_H_
#define _BILATERAL_H_

using namespace cv;

enum girdIndex {
	pixSum = 0,  //双边网格顶点包含像素点数
	lables = 1,  //双边网格顶点的标签（分割结果）
	bothB_F = 2,  //双边网格顶点是否同时有前背景点
	vIdx = 3,   //图割时的顶点
};

class Bilateral
{
public:
	std::vector<Mat> imgSrcArr,maskArr;	 //输入图片数据
	std::vector<Mat> bgModelArr, fgModelArr;	//前背景高斯模型
	Mat grid,gridColor;	//升维，平均取点，得到的grid。6维数组，保存顶点值与邻近像素点总数。
	const int gridSize[6] = { 3,30,50,16,16,16 };	//grid各个维度的大小,按顺序来为：t,x,y,r,g,b。
public:
	Bilateral(std::vector<Mat> imgSrcArr, std::vector<Mat> maskArr);
	~Bilateral();
	void run(int times);
private:
	void initGrid();
	void InitGmms();
	void nextGMMs();
	void constructGCGraph(GCGraph<double>& graph);
	int calculateVtxCount();
	void estimateSegmentation(GCGraph<double>& );
	void getGridPoint(int , const Point , int *, int , int , int );
	void getGridPoint(int , const Point , std::vector<int>& , int , int , int );
	void getMask();
};

#endif