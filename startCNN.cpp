#include "cnn.cpp"
#include "ReadData.h"

#define TRAIN 1//用来标识是训练样例还是测试样例
#define TEST 0

class CNN  m_cnn;//生成CNN的一个实例m_cnn

double LearningRate=0.001;//学习效率初始化
UCHR **image=NULL,*label=NULL;//样例及标签，全局变量
UCHR **myimage=NULL,*mylabel=NULL;//样例及标签，全局变量

void  startTraining();
void  startTest();

int main()
{

	srand((unsigned)time(NULL));
	image=readImage(TRAIN);//读入训练样例
	label=readLabel(TRAIN);//读入训练标签
    printf("read input data\n");

	myimage=readImage(TEST);//读入测试样例
	mylabel=readLabel(TEST);//读入测试标签
    printf("read label data\n");

    printf("before training \n");
	startTraining();//开始训练
    printf("after training \n");
	free(image);free(label);//释放空间
	free(myimage);free(mylabel);//释放空间

	//image=readImage(TEST);//读入测试样例
	//label=readLabel(TEST);//读入测试标签
	//startTest();//开始测试
	//free(image);free(label);//释放空间

	return 0;
}


///////////////////////////////////////////////////
int  Calculate(int index)
////////////////////////////////////////////////////
{
	int cCount = g_cImageSize*g_cImageSize;
	int ii, jj;
    double inputVector[(g_cImageSize+1)*(g_cImageSize+1)];//输入向量，扩展到29*29，并变换到[-1，1]
	//copy gray scale image to a double input vector in -1 to 1 range
	//-1 is white, 1 is black
	for ( ii=0; ii<g_cVectorSize * g_cVectorSize; ++ii ) inputVector[ii] = -1.0;
	for ( ii=0; ii<g_cImageSize; ++ii )
	{
		for ( jj=0; jj<g_cImageSize; ++jj )
		{
			int idxVector = 1 + jj + g_cVectorSize * (1 + ii);//扩充为29*29后的下标
			int idxImage = jj + g_cImageSize * ii;//28*28时的下标

			inputVector[ idxVector ] = (double)(2/255.0)*(int)(image[index][ idxImage ]) - 1.0;
		}
	}
	//call forward propagation function of CNN
	return m_cnn.Calculate(inputVector);
}

void random(int *p,int count)
{
	for(int i=0;i<count;i++)
	{
	    int n=rand()%count;
		int temp;
		temp=p[i];p[i]=p[n];p[n]=temp;
	}

}
//////////////////////////////////////////////////////
void  startTraining()
//////////////////////////////////////////////////////
{
	int errorCount;//记录训练的错误总数

	int N=84;//训练趟数

	//clock_t start,finish;

	double desiredOutputVector[g_cOutputSize];//实际输出
	int m_iOutput;//记录每个样例前向传播后的预测的值

	char outFileName[]="weights.txt";//保存权值的文本

	int index[g_cCountTrainingSample];//借助该数组把训练样例初始化
	for(int n=0;n<g_cCountTrainingSample;n++)  index[n]=n;//赋初值

	m_cnn.LoadWeightsRandom();//[-0.05,0.05]内的随机数
	//m_cnn.LoadWeights(outFileName);

    for(int k=1;k<=N;k++)
	{
		random(index,g_cCountTrainingSample);
		errorCount = 0;//每趟训练之前初始化为0
		for(int i=0; i<g_cCountTrainingSample; i++)
		{
			//前向传播
			m_iOutput=Calculate(index[i]);//第i个样例前向计算，返回本次样例的预测值
			if(m_iOutput != (int)label[index[i]]) errorCount++;//若预测值与实际值不相等，则训练错误总数加1
			//后向传播
			for(int j=0; j<g_cOutputSize; j++) desiredOutputVector[j] = -1;
			desiredOutputVector[(int)label[index[i]]] = 1;

			m_cnn.BackPropagate( desiredOutputVector, LearningRate );

			if(0==i%1000) printf("%d,%d\n",i,errorCount);
		}

		printf("第 %d 趟训练的错误数为：%d,正确率为: %f%%\n",k,errorCount,(1-(errorCount/60000.0))*100);

		FILE *fp;
		fp=fopen("error.txt","a");
		fprintf(fp,"第 %d 趟训练的错误数为：%d,正确率为: %f%%\n",k,errorCount,(1-(errorCount/60000.0))*100);
		fclose(fp);
		//调整学习效率
		if(k%6==0) LearningRate *= 0.794;
		//保存权值
		if(outFileName != NULL) m_cnn.SaveWeights(outFileName);//保存权值到outFileName文件
		//每趟训练，都测试一次
		startTest();//开始测试

	}


}

void startTest()
{
	int errCount=0;//记录测试总的错误个数
	char outFileName[]="weights.txt";//保存权值的文本
	int err;//记录每个样例是否分类正确

	m_cnn.LoadWeights(outFileName);//加载已经训练好的权值

	for(int i=0;i<g_cCountTestingSample;i++)//开始测试
	{
		err=Calculate(i);
		if(err != (int)label[i]) errCount++;

		if(0==i%100) printf("%d,%d\n",i,errCount);

	}
	printf("本趟测试的错误数为：%d,正确率为: %f%%\n",errCount,(1-(errCount/10000.0))*100);
	FILE *fp;
	fp=fopen("error.txt","a");
	fprintf(fp,"本趟测试的错误数为：%d,正确率为: %f%%\n",errCount,(1-(errCount/10000.0))*100);
	fclose(fp);

}
