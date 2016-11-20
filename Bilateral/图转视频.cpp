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
	g_imgSrc = imread("image/mallard-fly/00001.jpg");
	videowriter.open("image/mallard-fly.avi", CV_FOURCC('D', 'I', 'V', 'X'),24, Size(g_imgSrc.cols, g_imgSrc.rows));
	for (int i = 0; i < 10; i++)
	{
		std::stringstream s;
		std::string str;
		s <<"image/mallard-fly/0000"<< i << ".jpg";
		s >> str;
		std::cout << str << std::endl;
		g_imgSrc = imread(str);
		videowriter << g_imgSrc;
	}
	for (int i = 10; i < 80; i++)
	{
		std::stringstream s;
		std::string str;
		s << "image/mallard-fly/000" << i << ".jpg";
		s >> str;
		std::cout << str << std::endl;
		g_imgSrc = imread(str);
		videowriter << g_imgSrc;
	}
	videowriter.release();
}