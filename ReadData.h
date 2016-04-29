UCHR ** readImage(UINT type)
//读出MNIST中的所有样例
//type用来表示是训练样例还是测试样例,1表示训练，0表示测试
{
	UCHR **myimage;//image为所有的样例
	char *pathImage;//样例路径
	UINT imageNumber;//样例个数
    FILE *fp;//文件指针

	//要读的文件路径及样例的个数
	if(type==1)//表示需要读出训练样例
	{
		pathImage="./MNIST/train-images-idx3-ubyte";

		imageNumber=g_cCountTrainingSample;
	}
	else if(type==0)//表示需要读出测试样例
	{
		pathImage="./MNIST/t10k-images-idx3-ubyte";
		imageNumber=g_cCountTestingSample;
	}
	//读入Image，存放到image[][]中
	fp=fopen(pathImage,"rb");
	if(fp==NULL) printf("Cannot open the image file!");

	myimage=(UCHR **)malloc(sizeof(UCHR *)*imageNumber);
    fseek(fp,16,SEEK_SET);//跳过文件头16个字节
	for(UINT i=0;i<imageNumber;i++)
	{
		myimage[i]=(UCHR *)malloc(sizeof(UCHR)*g_cImageSize*g_cImageSize);
		fread(myimage[i],1,g_cImageSize*g_cImageSize,fp);//读入样例
	}
	fclose(fp);

	return myimage;
}
UCHR * readLabel(UINT type)
////读出MNIST中的所有标签
//type用来表示是训练样例还是测试样例,1表示训练，0表示测试
{
	UCHR *mylabel;//label为所有的标签
	char *pathLabel;//标签路径
	UINT imageNumber;//样例个数
    FILE *fp;//文件指针

	//要读的文件路径及样例的个数
	if(type==1)//表示需要读出训练样例
	{
		pathLabel="./MNIST/train-labels-idx1-ubyte";
		imageNumber=g_cCountTrainingSample;
	}
	else if(type==0)//表示需要读出测试样例
	{
		pathLabel="./MNIST/t10k-labels-idx1-ubyte";
		imageNumber=g_cCountTestingSample;
	}
	//读入标签，存入label[]中
	fp=fopen(pathLabel,"rb");
	if(fp==NULL)
	{
		printf("Cannot open the label file!");
	}

    mylabel=(UCHR *)malloc(sizeof(UCHR)*imageNumber);
    fseek(fp,8,SEEK_SET);//跳过文件头8个字节
	fread(mylabel,1,imageNumber,fp);//读入标签

	fclose(fp);

	return mylabel;
}
