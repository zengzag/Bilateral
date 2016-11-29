#pragma once
#include "opencv2/imgproc.hpp"
#include "GMM.h"
#include "gcgraph.hpp"

#ifndef _BILATERAL_H_
#define _BILATERAL_H_

using namespace cv;

enum girdIndex {
	pixSum = 0,  //���ص���
	fgdSum = 1,  //ǰ���������ڽ���ֵ��
	bgdSum = 2,  //����
	vIdx = 3   //�����ǩ
};

class Bilateral
{
public:
	std::vector<Mat> imgSrcArr;	 //����ͼƬ����
	Mat bgModel, fgModel;	//ǰ������˹ģ��
	Mat grid;	//��ά��ƽ��ȡ�㣬�õ���grid��6ά���飬���涥��ֵ���ڽ����ص�������
	const int gridSize[6] = { 3,30,50,16,16,16 };	//grid����ά�ȵĴ�С,��˳����Ϊ��t,x,y,r,g,b��
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