#include <iostream>
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <io.h>

using namespace::cv;
using namespace::std;

void getFiles(string path, string exd, vector<string>& files)
{
	//�ļ����
	intptr_t   hFile = 0;
	//�ļ���Ϣ
	struct _finddata_t fileinfo;
	string pathName, exdName;

	if (0 != strcmp(exd.c_str(), ""))
	{
		exdName = "\\*." + exd;
	}
	else
	{
		exdName = "\\*";
	}

	if ((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(), &fileinfo)) != -1)
	{
		do
		{
			//������ļ����������ļ���,����֮
			//�������,�����б�
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(pathName.assign(path).append("\\").append(fileinfo.name), exd, files);
			}
			else
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					files.push_back(pathName.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

int main() {

	vector<string> files;
	std::string name = "horse";
	std::string filePath = "E:\\Projects\\OpenCV\\DAVIS-data\\jump-dataset\\VIDEO-SNAPCUT\\" + name;
	getFiles(filePath, "jpg", files);
	
	VideoWriter videowriter;
	std::string	strVideowriter = "E:\\Projects\\OpenCV\\DAVIS-data\\jump-dataset\\" + name + ".avi";
	Mat g_imgSrc = imread(files[0]);
	videowriter.open(strVideowriter, CV_FOURCC('D', 'I', 'V', 'X'),24, Size(g_imgSrc.cols, g_imgSrc.rows));

	int size = files.size();
	for (int i = 0; i < size; i++)
	{		
		g_imgSrc = imread(files[i]);
		videowriter << g_imgSrc;
	}

	videowriter.release();
}