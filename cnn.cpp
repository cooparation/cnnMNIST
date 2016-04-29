#include "cnn.h"

CNN::CNN(void)
{
	ConstructNN();
}

CNN::~CNN(void)
{
	DeleteNN();
}

////////////////////////////////////
void CNN::ConstructNN()
////////////////////////////////////
{
	int i;
	m_nLayer = 5;//该CNN含有5层，利用这个信息可知第5层即为输出层

	m_Layer = new Layer[m_nLayer];//生成m_nLayer个层所需的空间

    //对生成的各层进行初始化
	m_Layer[0].pLayerPrev = NULL;
	for(i=1; i<m_nLayer; i++) m_Layer[i].pLayerPrev = &m_Layer[i-1];

	m_Layer[0].Construct	(	INPUT_LAYER,		1,		29,		0,	0	);//输入层
	m_Layer[1].Construct	(	CONVOLUTIONAL,		6,		13,		5,	2	);//卷积层
	m_Layer[2].Construct	(	CONVOLUTIONAL,		50,		5,		5,	2	);//卷积层
	m_Layer[3].Construct	(	FULLY_CONNECTED,	100,	1,		5,	1	);//全连接层
	m_Layer[4].Construct	(	FULLY_CONNECTED,	10,		1,		1,	1	);//全连接层
}

///////////////////////////////
void CNN::DeleteNN()
///////////////////////////////
{
	for(int i=0; i<m_nLayer; i++) m_Layer[i].Delete();
}

//////////////////////////////////////////////
void CNN::LoadWeightsRandom()
/////////////////////////////////////////////
//随机化装载权值，这些权值服从N(0,0.05^2)的分布
{
	int i, j, k, m;

	srand((unsigned)time(0));

	for ( i=1; i<m_nLayer; i++ )
	{
		for( j=0; j<m_Layer[i].m_nFeatureMap; j++ )
		{
			m_Layer[i].m_FeatureMap[j].bias = 0.05 * RANDOM_PLUS_MINUS_ONE;//偏置bias初始化

			for(k=0; k<m_Layer[i].pLayerPrev->m_nFeatureMap; k++)
				for(m=0; m < m_Layer[i].m_KernelSize * m_Layer[i].m_KernelSize; m++)
					m_Layer[i].m_FeatureMap[j].kernel[k][m] = 0.05 * RANDOM_PLUS_MINUS_ONE;//kernel初始化
		}
	}
}

//////////////////////////////////////////////
void CNN::LoadWeights(char *FileName)
/////////////////////////////////////////////
//从FileName所指文件中装载权值，和该程序中保存权值的格式是一样的
{
	int i, j, k, m;

	FILE *fp;
	if((fp = fopen(FileName, "r")) == NULL) return;

	for ( i=1; i<m_nLayer; i++ )
	{
		for( j=0; j<m_Layer[i].m_nFeatureMap; j++ )
		{
			fscanf(fp, "%lg ", &m_Layer[i].m_FeatureMap[j].bias);

			for(k=0; k<m_Layer[i].pLayerPrev->m_nFeatureMap; k++)
				for(m=0; m < m_Layer[i].m_KernelSize * m_Layer[i].m_KernelSize; m++)
					fscanf(fp, "%lg ", &m_Layer[i].m_FeatureMap[j].kernel[k][m]);
		}
	}
	fclose(fp);
}

//////////////////////////////////////////////
void CNN::SaveWeights(char *FileName)
/////////////////////////////////////////////
//保存权值到FileName所指的文件中
{
	int i, j, k, m;

	FILE *fp;
	if((fp = fopen(FileName, "w")) == NULL) return;

	for ( i=1; i<m_nLayer; i++ )
	{
		for( j=0; j<m_Layer[i].m_nFeatureMap; j++ )
		{
			fprintf(fp, "%lg ", m_Layer[i].m_FeatureMap[j].bias);

			for(k=0; k<m_Layer[i].pLayerPrev->m_nFeatureMap; k++)
				for(m=0; m < m_Layer[i].m_KernelSize * m_Layer[i].m_KernelSize; m++)
				{
					fprintf(fp, "%lg ", m_Layer[i].m_FeatureMap[j].kernel[k][m]);
				}
		}
	}

	fclose(fp);
}

//////////////////////////////////////////////////////////////////////////
int CNN::Calculate(double *input)
//////////////////////////////////////////////////////////////////////////
//前向传播，需要给出输入；该函数返回input输入情况下的预测输出
{
	int i, j;
    double output[g_cOutputSize];//保存输出层的输出
	//把input赋值给layer 0
	for(i=0; i<m_Layer[0].m_nFeatureMap; i++)
		for(j=0; j < m_Layer[0].m_FeatureSize * m_Layer[0].m_FeatureSize; j++)
							m_Layer[0].m_FeatureMap[0].value[j] = input[j];

	//前向传播(forward propagation),从第二层(Layer[1])开始，计算每层中每个神经元的输出
	for(i=1; i<m_nLayer; i++)
	{
		//每个神经元的输出或者说每个feature maps在前向传播之前都应初始化为0
		for(j=0; j<m_Layer[i].m_nFeatureMap; j++)
					m_Layer[i].m_FeatureMap[j].Clear();

		//调用层让它开始前向传播
		m_Layer[i].Calculate();
	}

	//前向传播结束时，把输出层赋值给output
	for(i=0; i<m_Layer[m_nLayer-1].m_nFeatureMap; i++) //这一步可以不要,可以直接比较输出层中的各输出,这样只是为了增加可读性
		output[i] = m_Layer[m_nLayer-1].m_FeatureMap[i].value[0];

	//找到最大输出的下标，并把它作为预测的输出
	j = 0;
	for(i=1; i<m_Layer[m_nLayer-1].m_nFeatureMap; i++)
		if(output[i] > output[j]) j = i;

	return j;//把输出层中最大的一个神经元，作为预测的输出
}

///////////////////////////////////////////////////////////
void CNN::BackPropagate(double *desiredOutput, double eta)
///////////////////////////////////////////////////////////
//反向传播函数，需要给出期望输出，和学习效率
{
	int i;
	//计算误差关于输出层输出的偏导数，即关于输出层X的偏导数
	for(i=0; i<m_Layer[m_nLayer-1].m_nFeatureMap; i++)
	{
		m_Layer[m_nLayer-1].m_FeatureMap[i].dError[0] =
		    m_Layer[m_nLayer-1].m_FeatureMap[i].value[0] - desiredOutput[i];
	}

	//从输出层开始，将上述偏导数反向传播，直到输入层的下一层(前向传播时的第一个隐藏层)为止
	for(i=m_nLayer-1; i>0; i--)
	{
		m_Layer[i].BackPropagate(eta);
	}

}

/////////////////////////////////////////////////////////////////////////////////////////////////////
void Layer::Construct(int type, int nFeatureMap, int FeatureSize, int KernelSize, int SamplingFactor)
/////////////////////////////////////////////////////////////////////////////////////////////////////
{
	m_type = type;//该层的类型
	m_nFeatureMap = nFeatureMap;//该层含有featureMap的个数
	m_FeatureSize = FeatureSize;//该层feature的大小
	m_KernelSize = KernelSize;//该层卷积窗口的大小
	m_SamplingFactor = SamplingFactor;//该层进行卷积时，窗口移动的步长；全连接可看成步长为1的卷积

	m_FeatureMap = new FeatureMap[ m_nFeatureMap ];//生成m_nFeatureMap个featureMap所需的空间

	for(int j=0; j<m_nFeatureMap; j++)//对这些featureMap进行初始化
	{
		m_FeatureMap[j].pLayer = this;
		m_FeatureMap[j].Construct(  );
	}
}

/////////////////////////
void Layer::Delete()
/////////////////////////
{
	for(int j=0; j<m_nFeatureMap; j++) m_FeatureMap[j].Delete();
}

///////////////////////////////////////////////////
void Layer::Calculate()
///////////////////////////////////////////////////
//Layer的前向传播，每个神经元都具有这种格式：Y->sigmoid->X
{
	for(int i=0; i<m_nFeatureMap; i++)//循环当前层的每个featureMap
	{
	    for(int k=0; k < m_FeatureSize * m_FeatureSize; k++)//先把第i个featureMap的bias加上，以后的步骤只需进行卷积计算即可
		{
				m_FeatureMap[i].value[k] = m_FeatureMap[i].bias;
		}
		int j;
		//调用featureMap进行卷积计算，只是把输入简单的相加，得到Y
		for( j=0; j<pLayerPrev->m_nFeatureMap; j++)//对前一层的每个featureMap卷积，计算当前层的第i个featureMap
		{
			m_FeatureMap[i].Calculate(
							pLayerPrev->m_FeatureMap[j].value,		//input feature map
							j										//index of input feature map
										);
		}

		//把Y经过SIGMOD函数挤压，得到该层的输出X
		for(j=0; j < m_FeatureSize * m_FeatureSize; j++)
		{
			m_FeatureMap[i].value[j] = SIGMOID(m_FeatureMap[i].value[j]);
		}
	}
}

///////////////////////////////////////////////////////////////
void Layer::BackPropagate(double etaLearningRate)
//////////////////////////////////////////////////////////////
//Layer进行反向传播
{
	int i,j;
	for(i=0; i<m_nFeatureMap; i++)
	{
		for(j=0; j < m_FeatureSize * m_FeatureSize; j++)
		{
			double temp = DSIGMOID(m_FeatureMap[i].value[j]);
			m_FeatureMap[i].dError[j] = temp * m_FeatureMap[i].dError[j];//开始存的是关于X的偏导数，乘以temp后变成
			                                                             //关于Y的偏导数
		}
	}

	for(i=0; i<m_nFeatureMap; i++)  m_FeatureMap[i].ClearDErrWrtW();//误差关于权值的偏导数初始化为0

	for(i=0; i<pLayerPrev->m_nFeatureMap; i++)//该层的前一层(前向传播方向)关于输出X的偏导数清0
		pLayerPrev->m_FeatureMap[i].ClearDError();

	//反向传播，误差反向传播到前一层
	for(i=0; i<m_nFeatureMap; i++)//对当前层的每个featureMap进行处理
	{
		for( j=0; j<pLayerPrev->m_nFeatureMap; j++)//计算当前featureMap对前一层各featureMap的影响，
		{                                          //循环当前层的每个kernel，相当于同时循环前一层的每个featureMap
			m_FeatureMap[i].BackPropagate(
				pLayerPrev->m_FeatureMap[j].value,	//前一层的feature map
				j,									//前一层第j个feature map
				pLayerPrev->m_FeatureMap[j].dError	//前一层误差关于X的偏导数
				);
		}

		for(int j=0; j<m_FeatureSize * m_FeatureSize; j++)//误差关于bias的偏导数
			m_FeatureMap[i].dErr_wrtB += m_FeatureMap[i].dError[j];
	}

	//更新权值
	for(i=0; i<m_nFeatureMap; i++)
	{
		m_FeatureMap[i].bias -= etaLearningRate * m_FeatureMap[i].dErr_wrtB;//更新bias

		for(int j=0; j<pLayerPrev->m_nFeatureMap; j++)//更新kernel
		{
			for(int k=0; k < m_KernelSize * m_KernelSize; k++)
				m_FeatureMap[i].kernel[j][k] -= etaLearningRate * m_FeatureMap[i].dErr_wrtW[j][k];
		}
	}

}


//////////////////////////////////////////////////////////////////////////////////////////
void FeatureMap::Construct( )
//////////////////////////////////////////////////////////////////////////////////////////
{
	int i;
	//前一层featureMap的个数
	if(pLayer->m_type == INPUT_LAYER) m_nFeatureMapPrev = 0;
	else m_nFeatureMapPrev = pLayer->pLayerPrev->m_nFeatureMap;

	int FeatureSize = pLayer->m_FeatureSize;//featureMap的大小
	int KernelSize  = pLayer->m_KernelSize;//卷积窗口的大小

	value = new double [ FeatureSize * FeatureSize ];//神经元的输出X
	dError = new double [ FeatureSize * FeatureSize ];//误差关于神经元输出X的偏导数

	//构造权值
	kernel = new double* [ m_nFeatureMapPrev ];
	for(i=0; i<m_nFeatureMapPrev; i++)
	{
		kernel[i] = new double [KernelSize * KernelSize];

		bias = 0.05 * RANDOM_PLUS_MINUS_ONE;//bias初始化
		for(int j=0; j < KernelSize * KernelSize; j++) kernel[i][j] = 0.05 * RANDOM_PLUS_MINUS_ONE;//kernel初始化
	}
	//误差关于权值的偏导数
	dErr_wrtW = new double* [ m_nFeatureMapPrev ];
	for(i=0; i<m_nFeatureMapPrev; i++)
		dErr_wrtW[i] = new double [KernelSize * KernelSize];

	//误差关于bias的偏导数dErr_wrtB无需初始化
}

///////////////////////////
void FeatureMap::Delete()
///////////////////////////
{
	delete[] value;
	delete[] dError;

	for(int i=0; i<m_nFeatureMapPrev; i++)
	{
		delete[] kernel[i];
		delete[] dErr_wrtW[i];
	}
}

////////////////////////////
void FeatureMap::Clear()
/////////////////////////////
{
	for(int i=0; i < pLayer->m_FeatureSize * pLayer->m_FeatureSize; i++) value[i] = 0.0;
}

////////////////////////////////
void FeatureMap::ClearDError()
/////////////////////////////////
{
	for(int i=0; i < pLayer->m_FeatureSize * pLayer->m_FeatureSize; i++) dError[i] = 0.0;
}

////////////////////////////////
void FeatureMap::ClearDErrWrtW()
////////////////////////////////
{
	dErr_wrtB = 0;
	for(int i=0; i < m_nFeatureMapPrev; i++)
		for(int j=0; j < pLayer->m_KernelSize * pLayer->m_KernelSize; j++) dErr_wrtW[i][j] = 0.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
void FeatureMap::Calculate(double *valueFeatureMapPrev, int idxFeatureMapPrev )
//////////////////////////////////////////////////////////////////////////////////////////////////////
//featureMap前向传播

//	valueFeatureMapPrev:指向前一层的一个featureMap的指针
//	idxFeatureMapPrev :	标识valueFeatureMapPrev是前一层的第几个featureMap
{
	int isize = pLayer->pLayerPrev->m_FeatureSize; //前一层featureMap的大小
	int ksize = pLayer->m_KernelSize;//当前层kernel窗口的大小
	int step_size = pLayer->m_SamplingFactor;//卷积的步长，全连接相当于步长为1的卷积

	int k = 0;

	for(int row0 = 0; row0 <= isize - ksize; row0 += step_size)
		for(int col0 = 0; col0 <= isize - ksize; col0 += step_size)
			value[k++] += Convolute(valueFeatureMapPrev, isize, row0, col0, kernel[idxFeatureMapPrev], ksize);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////
double FeatureMap::Convolute(double *input, int size, int r0, int c0, double *weight, int kernel_size)
//////////////////////////////////////////////////////////////////////////////////////////////////////
{
	int i, j, k = 0;
	double summ = 0;

	for(i = r0; i < r0 + kernel_size; i++)
		for(j = c0; j < c0 + kernel_size; j++)
			summ += input[i * size + j] * weight[k++];

	return summ;
}

/////////////////////////////////////////////////////////////////////////////////////
void FeatureMap::BackPropagate(double *valueFeatureMapPrev, int idxFeatureMapPrev,
							   double *dErrorFeatureMapPrev)
/////////////////////////////////////////////////////////////////////////////////////
//featureMap反向传播

//	valueFeatureMapPrev:前一层(前向传播方向)featureMap的输出
//	idxFeatureMapPrev :标识valueFeatureMapPrev保存的是前一层的第几个featureMap
//  dErrorFeatureMapPrev:前一层featureMap中，误差关于输出X的偏导数

{
	int isize = pLayer->pLayerPrev->m_FeatureSize;	//前一层featureMap的大小
	int ksize = pLayer->m_KernelSize;				//当前featureMap的kernel的大小
	int step_size = pLayer->m_SamplingFactor;		//当前层卷积窗口移动的步长

	int row0, col0, k;

	k = 0;
	for(row0 = 0; row0 <= isize - ksize; row0 += step_size)//循环得到的是前一层featureMap在卷积操作时，kernel窗口
	{                                                      //每次移动的起始位置
		for(col0 = 0; col0 <= isize - ksize; col0 += step_size)
		{
			for(int i=0; i<ksize; i++)
			{
				for(int j=0; j<ksize; j++)//前一层featureMap的每个起始位置的一个ksize*ksize区域对应当前层的一个神经元
				{
					//用于得到前一层误差关于X的偏导数
					double temp = kernel[idxFeatureMapPrev][i * ksize + j];
					dErrorFeatureMapPrev[(row0 + i) * isize + (j + col0)] += dError[k] * temp;

                    //用于得到当前层误差关于kernel的偏导数
					temp = valueFeatureMapPrev[(row0 + i) * isize + (j + col0)];
					dErr_wrtW[idxFeatureMapPrev][i * ksize + j] += dError[k] * temp;
				}//j
			}//i
			k++;//用来标识当前层的dError_wrt_Y，每个dError_wrt_Y对应前一层的一个ksize*ksize区域
		}//col0
	}//row0

}//end function
