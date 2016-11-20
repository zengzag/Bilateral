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
	Mat grid;	//升维，平均取点，得到的grid。6维数组，保存顶点值与邻近像素点总数。
	const int gridSize[6] = { 3,20,20,16,16,16 };	//grid各个维度的大小,按顺序来为：t,x,y,r,g,b。
	std::vector<std::vector<int> > grid_forePts;      //前景grid点
	std::vector<std::vector<int> > grid_backPts;      //背景grid点
public:
	Bilateral(std::vector<Mat> img);
	~Bilateral();
	void InitGmms(std::vector<Point>& forPts, std::vector<Point>& bacPts, int);
	void run(std::vector<Mat>& );
private:
	bool isPtInVector(Point pt, std::vector<Point>& points);
	void initGrid();
	void constructGCGraph(const GMM&, const GMM&, GCGraph<double>& graph);
	int calculateVtxCount();
	void estimateSegmentation(GCGraph<double>&, std::vector<Mat>& );
	void getGridPoint(int , const Point , int *);
	void getGridPoint(int , const Point , std::vector<int>& );
};

#endif