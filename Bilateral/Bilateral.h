#pragma once
#include "opencv2/imgproc.hpp"
#include "GMM.h"
#include "gcgraph.hpp"

#ifndef _BILATERAL_H_
#define _BILATERAL_H_

using namespace cv;

enum girdIndex {
	pixSum = 0,  //˫�����񶥵�������ص���
	lables = 1,  //˫�����񶥵�ı�ǩ���ָ�����
	bothB_F = 2,  //˫�����񶥵��Ƿ�ͬʱ��ǰ������
	vIdx = 3,   //ͼ��ʱ�Ķ���
};

class Bilateral
{
public:
	std::vector<Mat> imgSrcArr,maskArr;	 //����ͼƬ����
	std::vector<Mat> bgModelArr, fgModelArr;	//ǰ������˹ģ��
	Mat grid,gridColor;	//��ά��ƽ��ȡ�㣬�õ���grid��6ά���飬���涥��ֵ���ڽ����ص�������
	const int gridSize[6] = { 3,30,50,16,16,16 };	//grid����ά�ȵĴ�С,��˳����Ϊ��t,x,y,r,g,b��
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