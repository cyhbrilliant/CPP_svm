#include <iostream>
#include <opencv.hpp>
#include <ml.hpp>
#include <string>
#include <time.h>

using namespace std;
using namespace cv;
using namespace ml;

int NEGnum=0;
int POSnum=0;

char* NEGfile="NEG\\NEGLIST.txt";
char* POSfile="POS\\POSLIST.txt";
char* SVMClassifier="SVM_detector.xml";

int HOGweight=32;
int HOGheight=32;
int HOGstride=8;
int HOGblock=16;
int HOGcell=HOGblock/2;
int nBins=9;
int HOGvector=((HOGweight-HOGblock)/HOGstride+1)*((HOGheight-HOGblock)/HOGstride+1)*nBins*4; 


void HOGSVMBuild()
{
	HOGDescriptor hog(cvSize(HOGweight,HOGheight),cvSize(HOGblock,HOGblock),cvSize(HOGstride,HOGstride),cvSize(HOGcell,HOGcell),nBins,1,-1,HOGDescriptor::L2Hys,0.2,false,HOGDescriptor::DEFAULT_NLEVELS);           //初始化HOG描述符  

	FILE* NEGfs =fopen(NEGfile,"r"); 
	FILE* POSfs =fopen(POSfile,"r"); 

	char NEGimgname[1024];
	char POSimgname[1024];

	vector<float> NEGdescrip;  
	vector<float> POSdescrip;  

	vector<float> ALLdescrip;
	
	Mat data_mat;
	Mat res_mat;
	
	int POSnum=0;
	int NEGnum=0;

	while(POSfs)
	{
		if(!fgets(POSimgname, (int)sizeof(POSimgname)-2, POSfs))  
			break;  
		//while(*filename && isspace(*filename))  
		//  ++filename;  
		if(POSimgname[0] == '#')  
			continue;  
		int l = strlen(POSimgname);  
		while(l > 0 && isspace(POSimgname[l-1]))  
			--l;  
		POSimgname[l] = '\0';  

		char POSname[1024];
		sprintf(POSname,"POS\\%s",POSimgname);
		Mat POSimg = imread(POSname);  		
		//imshow(POSname,POSimg);
		//waitKey(0);
		resize(POSimg,POSimg,cvSize(HOGweight,HOGheight));
		hog.compute(POSimg,POSdescrip);
		POSnum++;
		for (int i=0;i<POSdescrip.size();i++)
		{
			ALLdescrip.push_back(POSdescrip.at(i));
		}
		POSdescrip.clear();
	}

	while(NEGfs)
	{
		if(!fgets(NEGimgname, (int)sizeof(NEGimgname)-2, NEGfs))  
			break;  
		//while(*filename && isspace(*filename))  
		//  ++filename;  
		if(NEGimgname[0] == '#')  
			continue;  
		int l = strlen(NEGimgname);  
		while(l > 0 && isspace(NEGimgname[l-1]))  
			--l;  
		NEGimgname[l] = '\0';  

		char NEGname[1024];
		sprintf(NEGname,"NEG\\%s",NEGimgname);
		Mat NEGimg = imread(NEGname);  		
		//imshow(NEGname,NEGimg);
		//waitKey(0);
		resize(NEGimg,NEGimg,cvSize(HOGweight,HOGheight));
		hog.compute(NEGimg,NEGdescrip);
		NEGnum++;
		for (int i=0;i<NEGdescrip.size();i++)
		{
			ALLdescrip.push_back(NEGdescrip.at(i));
		}
		NEGdescrip.clear();
	}

	res_mat=Mat::ones(POSnum+NEGnum,1,CV_32SC1);
	for (int i=POSnum;i<POSnum+NEGnum;i++)
	{
		res_mat.at<int>(i,0)=-1;
	}

	float* buf2=&ALLdescrip[0];  
	data_mat=Mat(POSnum+NEGnum,HOGvector,CV_32FC1,buf2);

	Ptr<SVM> svm = SVM::create();
	svm->setKernel(SVM::LINEAR);
	//svm->setCoef0(0.0);
	//svm->setDegree(0);
	//svm->setTermCriteria(TermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 1e-3 ));
	//svm->setGamma(1);
	//svm->setNu(0.5);
	//svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
	//svm->setC(0.01); // From paper, soft classifier
	svm->setType(ml::SVM::C_SVC); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
	svm->train(data_mat,ml::ROW_SAMPLE,res_mat);
	clog << "...[SVM done]" << endl;

	svm->save( SVMClassifier );

	fclose(NEGfs);
	fclose(POSfs);

}


void getSVM(vector<float> &classifi,String svmLoad)
{
	Ptr<SVM> svm=Algorithm::load<SVM>(svmLoad);
	Mat supportVec=svm->getSupportVectors();

	int svNUM=svm->getVarCount();

	Mat alpha;
	Mat svidx;
	double rho;
	rho=svm->getDecisionFunction(0,alpha,svidx);
	Mat alphaFl(alpha.rows,alpha.cols,CV_32FC1);
	for (int i=0;i<alpha.rows;i++)
	{
		for (int j=0;j<alpha.cols;j++)
		{
			alphaFl.at<float>(i,j)=(float)alpha.at<double>(i,j);
		}
	}

	Mat cowSVfl=-1.0*alphaFl*supportVec;

	classifi.clear();
	for (int i=0;i<cowSVfl.cols;i++)
	{
		classifi.push_back(cowSVfl.at<float>(0,i));
	}

	classifi.push_back(float(rho));
}


void DetectOBJ(Mat image,vector<float> &classifi,std::vector<cv::Rect> &regions)
{
	regions.clear();
	int weight=HOGweight;
	int height=HOGheight;
	HOGDescriptor hog(cvSize(weight,height),cvSize(16,16),cvSize(8,8),cvSize(8,8),9,1,-1,HOGDescriptor::L2Hys,0.2,false,HOGDescriptor::DEFAULT_NLEVELS);        

	hog.setSVMDetector(classifi);

	// 3. 在测试图像上检测行人区域
	
	hog.detectMultiScale(image, regions, 0.0, cv::Size(8,8), cv::Size(0,0), 1.05, 1);


}

void NEGlearning(Mat Frame,std::vector<cv::Rect> regions)
{
	FILE* NEGfs =fopen(NEGfile,"a+"); 
	char filename[200];
	for (int i=0;i<regions.size();i++)
	{
		Mat HardExample;
		Frame(regions.at(i)).copyTo(HardExample);
		resize(HardExample,HardExample,cvSize(HOGweight,HOGheight));

		long fileclock=clock();
		sprintf(filename,"NEG\\%ld.jpg",fileclock);
		imwrite(filename,HardExample);

		fprintf(NEGfs,"\n%ld.jpg",fileclock);
	}

	fclose(NEGfs);
}

void drawOBJ(Mat image,std::vector<cv::Rect> regions)
{
	// 显示
	for (size_t i = 0; i < regions.size(); i++)
	{
		cv::rectangle(image, regions[i], cv::Scalar(0,0,255), 2);
		cout<<"x:"<<regions[i].x<<"  y:"<<regions[i].y<<endl;
	}
}







void main()
{
	
	HOGSVMBuild();

	vector<float> svmFL;
	getSVM(svmFL,SVMClassifier);
	std::vector<cv::Rect> regions;

	Mat frame;



	VideoCapture cap(1);
	while (true)
	{
		//===
		//OrbDetect_data.clear();
		//OrbKeypoint.clear();
		//===

		cap.read(frame);
		resize(frame,frame,cvSize(320,240));


		DetectOBJ(frame,svmFL,regions);

	/*	if (!regions.empty())
		{
			NEGlearning(frame,regions);
			HOGSVMBuild();
			getSVM(svmFL,SVMClassifier);
		}*/
















		drawOBJ(frame,regions);

		
		imshow("show",frame);

		waitKey(10);

	}


	system("pause");

}

