#include "global.h"

//typedef unsigned char uchar;
#define INPUT_LAYER		0
#define CONVOLUTIONAL	1
#define FULLY_CONNECTED 2

#define max(a,b) ((a)>(b) ? (a):(b))
#define RANDOM_PLUS_MINUS_ONE	( (double)(2.0 * rand())/RAND_MAX - 1.0 )	//[-1,1]的随机数
#define SIGMOID(x) (1.7159*tanh(0.66666667*(x)))
#define DSIGMOID(x) (0.66666667/1.7159*(1.7159+(x))*(1.7159-(x)))


class Layer;

class FeatureMap
{
public:
	double bias, dErr_wrtB;//偏置，误差关于偏置的偏导数
	double *value, *dError;//每个神经元的输出，误差关于输出的偏导数
	double **kernel, **dErr_wrtW;//权值，误差关于权值的偏导数

	int m_nFeatureMapPrev;//前一层featureMap的大小

	Layer *pLayer;//指向featureMap所在层

	void Construct( );
	void Delete();

	void Clear();
	void ClearDError();
	void ClearDErrWrtW();

	double Convolute(double *input, int size, int r0, int c0, double *weight, int kernel_size);
	void Calculate(double *valueFeatureMapPrev, int idxFeatureMapPrev );
	void BackPropagate(double *valueFeatureMapPrev, int idxFeatureMapPrev, double *dErrorFeatureMapPrev);
};

class Layer
{
public:
	int m_type;//标识该层的类型，是输入层、卷积层还是全连接层
	int m_SamplingFactor;//该层中kernel窗口移动的步长，它是和m_type相对应的,输入层取0，卷积层取2，全连接层取1

	Layer *pLayerPrev;//指向当前层的前一层

	int m_nFeatureMap;//当前层featureMap的个数
	int m_FeatureSize;//当前层featureMap的大小
	int m_KernelSize;//当前层kernel的大小

	FeatureMap* m_FeatureMap;//指向当前层的指针

	void ClearAll()
	{
		for(int i=0; i<m_nFeatureMap; i++)
		{
			m_FeatureMap[i].Clear();
			m_FeatureMap[i].ClearDError();
			m_FeatureMap[i].ClearDErrWrtW();
		}
	}

	void Calculate();
	void BackPropagate(double etaLearningRate);

	void Construct(int type, int nFeatureMap, int FeatureSize, int KernelSize, int SamplingFactor);
	void Delete();
};

class CNN
{
public:
	CNN(void);
	~CNN(void);

	Layer *m_Layer;//指向当前神经网络的Layer的指针
	int m_nLayer;//该神经网络所含层数

	void ConstructNN();
	void DeleteNN();

	void LoadWeights(char *FileName);
	void LoadWeightsRandom();
	void SaveWeights(char *FileName);
	int Calculate(double *input);
	void BackPropagate(double *desiredOutput, double eta);
};
