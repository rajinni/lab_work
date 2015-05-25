// svmtest.cpp : Defines the entry point for the console application.
//

#define _CRT_SECURE_NO_WARNINGS
#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>

//定义动态二维数组用，必检查
#define dimension 50
#define studentnum 114
#define trainnum 57
#define testnum 57
#define prediction_mode// means using class 1 as training class 2 as prediction
//#undef  prediction_mode
using namespace cv;
using namespace std;

std::vector<float>dimen;
std::vector<float>target;
std::vector<float>check;
std::vector<float>result;
//std::vector<struct num>vec;
vector<float>::iterator iter1;
float Tp[5] = { 0, 0, 0, 0, 0 };
float Tn[5] = { 0, 0, 0, 0, 0 };
float Fp[5] = { 0, 0, 0, 0, 0 };
float Fn[5] = { 0, 0, 0, 0, 0 };
float vecs[studentnum][dimension];
float trainvecs[trainnum][dimension];
float testvecs[testnum][dimension];
float averageaccuracy = 0;
//int m = dimension;
//int n = studentnum;
//vector<vector<int> >vecInt(n, vector<int>(m));
//for (int i = 0; i < n; i++)    //初始化二维数组，，其实这里可以不用初始化的，vector中默认初始化为0
//for (int j = 0; j < m; j++)
//	vecInt[i][j] = 0;

int main()
{
	// step 1:  
	//训练数据的分类标记，即4类
	char buffer[1000];
	int t = 0;
	ifstream infile1("15_50svm.data");
	while (!infile1.std::ios::eof())
	{
		infile1.getline(buffer, 1000);
		char* a;
		a = strtok(buffer, " ");
		while (a != NULL)
		{
			float b = atof(a);
			dimen.push_back(b);
			a = strtok(NULL, " ");
		}
		//输入每个vector到dimen中


		for (int j = 0; j < dimen.size(); j++)
		{
			vecs[t][j] = dimen[j];
			//			cout << vecs[t][j] << " ";
		}

		//		t = t + 1;
		//		cout << endl;

	}

	for (int x = 0; x < trainnum; x++)
	{
		for (int y = 0; y < dimension; y++)
		{
			trainvecs[x][y] = vecs[x][y];
		}
	}
	for (int x = 0; x < testnum; x++)
	{
		for (int y = 0; y < dimension; y++)
		{
			testvecs[x][y] = vecs[x + trainnum][y];
		}
	}
	//	cout << &trainvecs;
	std::fstream fin("target15.txt");
	while (!fin.std::ios::eof())
	{
		float a;
		fin >> a;
		target.push_back(a);
	}
	CvMat labelsMat;
#ifdef prediction_mode
	{
		float labels[trainnum];
		for (int f = 0; f < trainnum; f++)
		{
			labels[f] = target[f];
		}
		for (int g = 0; g < testnum; g++)
		{
			check.push_back(target[g + trainnum]);
		}
		labelsMat = cvMat(trainnum, 1, CV_32FC1, labels);
	}
#else
	{
		float labels[testnum];
		for (int f = 0; f < testnum; f++)
		{
			labels[f] = target[f+trainnum];
		}
		for (int g = 0; g < trainnum; g++)
		{
			check.push_back(target[g]);
		}
		labelsMat = cvMat(testnum, 1, CV_32FC1, labels);
	}
#endif
	
	
	//训练数据矩阵
	//cout << &labelsMat;
	CvMat trainingDataMat;
#ifdef prediction_mode
	{
		trainingDataMat = cvMat(trainnum, dimension, CV_32FC1, trainvecs);
	}
#else
	trainingDataMat = cvMat(testnum, dimension, CV_32FC1, testvecs);	
#endif
	//	CvMat trainingDataMat = cvMat(trainnum, dimension, CV_32FC1, trainvecs);
	//cout << & trainingDataMat;
	// step 2:  
	//训练参数设定
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;				 //SVM类型
	params.kernel_type = CvSVM::RBF;			 //核函数的类型
	params.C = 500;
	params.gamma = 1;
	//SVM训练过程的终止条件, max_iter:最大迭代次数  epsilon:结果的精确性
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON);

	// step 3:  
	//启动训练过程
	CvSVM SVM;
	//	cout << &trainingDataMat;

	SVM.train(&trainingDataMat, &labelsMat, NULL, NULL, params);

	float a[dimension];
	// step 4:  
	//使用训练所得模型对新样本进行分类测试
#ifdef prediction_mode
	{
		std::ofstream outfile("result.csv", std::ios::trunc);
		for (int i = 0; i<testnum; i++)
		{
			for (int j = 0; j<dimension; j++)
			{

				a[j] = testvecs[i][j];

			}
			CvMat sampleMat;
			cvInitMatHeader(&sampleMat, 1, dimension, CV_32FC1, a);
			//		cvmSet(&sampleMat, 0, 0, i);										// Set M(i,j)
			//		cvmSet(&sampleMat, 0, 1, j);										// Set M(i,j)
			float response = SVM.predict(&sampleMat);
			result.push_back(response);
			outfile << response << "\n";
			//		cout << endl;
		}
	}
#else
	{
		std::ofstream outfile("result.csv", std::ios::trunc);
		for (int i = 0; i<trainnum; i++)
		{
			for (int j = 0; j<dimension; j++)
			{

				a[j] = trainvecs[i][j];

			}
			CvMat sampleMat;
			cvInitMatHeader(&sampleMat, 1, dimension, CV_32FC1, a);
			//		cvmSet(&sampleMat, 0, 0, i);										// Set M(i,j)
			//		cvmSet(&sampleMat, 0, 1, j);										// Set M(i,j)
			float response = SVM.predict(&sampleMat);
			result.push_back(response);
			outfile << response << "\n";
			//		cout << endl;
		}
	}
#endif
	for (int i = 1; i <= 5; i++)
	{
		for (int m = 0; m != result.size(); m++)
		{
			if (check[m] == i && result[m] == i)
			{
				Tp[i - 1]++;
			}
			if (check[m] == i && result[m] != i)
			{
				Tn[i - 1]++;
			}

			if (check[m] != i && result[m] == i)
			{
				Fp[i - 1]++;
			}
			if (check[m] != i && result[m] != i)
			{
				Fn[i - 1]++;
			}

		}
	}
	std::cout << Tp[0] << "," << Tp[1] << "," << Tp[2] << "," << Tp[3] << "," << Tp[4] << std::endl;
	std::cout << Tn[0] << "," << Tn[1] << "," << Tn[2] << "," << Tn[3] << "," << Tn[4] << std::endl;
	std::cout << Fp[0] << "," << Fp[1] << "," << Fp[2] << "," << Fp[3] << "," << Fp[4] << std::endl;
	std::cout << Fn[0] << "," << Fn[1] << "," << Fn[2] << "," << Fn[3] << "," << Fn[4] << std::endl;
	float accuracy=0;
	for (int i = 0; i < 5; i++)
	{
		accuracy = (Tp[i] + Fn[i]) / (Tp[i] + Tn[i] + Fp[i] + Fn[i]);
		averageaccuracy += accuracy;
	}
	averageaccuracy = averageaccuracy / 5;
	cout << averageaccuracy << endl;
//	CvMat sampleMat;
//	cvInitMatHeader(&sampleMat, testnum, dimension, CV_32FC1, testvecs);
//	float response = SVM.predict(&sampleMat);
//	cout << response << " ";
	// step 5:  
	//获取支持向量
	int c = SVM.get_support_vector_count();
	cout << endl;
	for (int i = 0; i<c; i++)
	{
		const float* v = SVM.get_support_vector(i);
		cout << *v << " ";
	}
	cout << endl;

	system("pause");
	return 0;
}