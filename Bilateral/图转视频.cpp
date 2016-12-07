#include <iostream>
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include <sstream>

using namespace::cv;

int main() {
	VideoWriter videowriter;
	Mat g_imgSrc;

	std::string name = "123";
	std::string strPmgSrc = "monkeydog/Comp_195.bmp";
	std::string	strVideowriter = "E:/Projects/OpenCV/DAVIS-data/image/" + name + ".avi";

	g_imgSrc = imread(strPmgSrc);
	videowriter.open(strVideowriter, CV_FOURCC('D', 'I', 'V', 'X'),24, Size(g_imgSrc.cols, g_imgSrc.rows));
	for (int i = 195; i < 266; i++)
	{
		std::stringstream s;
		std::string str;
		s <<"monkeydog/Comp_"<< i << ".bmp";
		s >> str;
		std::cout << str << std::endl;
		g_imgSrc = imread(str);
		videowriter << g_imgSrc;
	}
	/*for (int i = 10; i < 99; i++)
	{
		std::stringstream s;
		std::string str;
		s << "E:/Projects/OpenCV/DAVIS-data/DAVIS/JPEGImages/480p/" << name << "/000" << i << ".jpg";
		s >> str;
		std::cout << str << std::endl;
		g_imgSrc = imread(str);
		videowriter << g_imgSrc;
	}*/
	videowriter.release();
}