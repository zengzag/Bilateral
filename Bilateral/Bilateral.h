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
	Mat imgSrc;		//����ͼƬ����
	Mat bgModel, fgModel;	//ǰ������˹ģ��
	Mat grid;	//��ά��ƽ��ȡ�㣬�õ���grid��6ά���飬���涥��ֵ���ڽ����ص�������
	const int gridSize[6] = { 1,40,40,32,32,32 };	//grid����ά�ȵĴ�С,��˳����Ϊ��t,x,y,r,g,b��
public:
	Bilateral(Mat& img);
	~Bilateral();
	void InitGmms(std::vector<Point>& forPts, std::vector<Point>& bacPts);
	void run();
private:
	bool isPtInVector(Point pt, std::vector<Point>& points);
	void initGrid();
};

#endif