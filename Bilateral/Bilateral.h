#pragma once
#include "opencv2/imgproc.hpp"
#include "GMM.h"
#include "gcgraph.hpp"

#ifndef _BILATERAL_H_
#define _BILATERAL_H_

using namespace cv;

enum girdIndex {
	pixSum = 0,  //像素点数
	fgdSum = 1,  //前景点数（邻近插值）
	bgdSum = 2,  //背景
	vIdx = 3   //顶点标签
};

class Bilateral
{
public:
	std::vector<Mat> imgSrcArr;	 //输入图片数据
	Mat bgModel, fgModel;	//前背景高斯模型
	Mat grid;	//升维，平均取点，得到的grid。6维数组，保存顶点值与邻近像素点总数。
	const int gridSize[6] = { 3,30,50,16,16,16 };	//grid各个维度的大小,按顺序来为：t,x,y,r,g,b。
public:
	Bilateral(std::vector<Mat> img);
	~Bilateral();
	void InitGmms(Mat& , int);
	void run(std::vector<Mat>& );
private:
	void initGrid();
	void constructGCGraph(const GMM&, const GMM&, GCGraph<double>& graph);
	int calculateVtxCount();
	void estimateSegmentation(GCGraph<double>&, std::vector<Mat>& );
	void getGridPoint(int , const Point , int *, int , int , int );
	void getGridPoint(int , const Point , std::vector<int>& , int , int , int );
};

#endif