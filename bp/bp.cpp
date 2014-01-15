
#include <stdafx.h>
#include<iostream>
#include<cmath>
#include<time.h>
#include<Windows.h>

using namespace std;

#define  innode 437  //输入结点数
#define  hidenode 29//隐含结点数
#define  outnode 26 //输出结点数
#define  trainsample 190//BP训练样本数
#define  testsample 190 //测试样本数
#define  digit_width 19 //每个字符宽度
#define  digit_height 23 //每个字符高度
#define  target_error  0.1 //误差指标
class BpNet
{
public:
    void train(double p[trainsample][innode ],double t[trainsample][outnode]);//Bp训练
    double p[trainsample][innode];     //训练输入
    double t[trainsample][outnode];    //训练输出

    double *recognize(double *p);//Bp识别

    void writetrain(); //保存训练好的权值
    void readtrain(); //读训练好的权值，这使的不用每次去训练了，只要把训练最好的权值存下来就OK

    BpNet();
    virtual ~BpNet();

public:
    void init();
    double w[innode][hidenode];//隐含结点权值
    double w1[hidenode][outnode];//输出结点权值
    double b1[hidenode];//隐含结点阀值
    double b2[outnode];//输出结点阀值

    double rate_w; //权值学习率（输入层-隐含层)
    double rate_w1;//权值学习率 (隐含层-输出层)
    double rate_b1;//隐含层阀值学习率
    double rate_b2;//输出层阀值学习率

    double e;//误差计算
    double error;//允许的最大误差
    double result[outnode];// Bp输出
};

BpNet::BpNet()
{
    error=1.0;
    e=0.0;

    rate_w=0.95;  //权值学习率（输入层--隐含层)
    rate_w1=0.95; //权值学习率 (隐含层--输出层)
    rate_b1=0.95; //隐含层阀值学习率
    rate_b2=0.95; //输出层阀值学习率
}

BpNet::~BpNet()
{

}

void winit(double w[],int n) //权值初始化
{
  for(int i=0;i<n;i++)
    w[i]=(2*(double)rand()/(RAND_MAX))-1;
}

void BpNet::init()
{
    winit((double*)w,innode*hidenode);
    winit((double*)w1,hidenode*outnode);
    winit(b1,hidenode);
    winit(b2,outnode);
}

/*BP网络的单次训练函数，传输层和隐含层均为logsig函数*/
void BpNet::train(double p[trainsample][innode],double t[trainsample][outnode])
{
    double pp[hidenode];//隐含结点的校正误差
    double qq[outnode];//希望输出值与实际输出值的偏差
    double yd[outnode];//希望输出值

    double x[innode]; //输入向量
    double x1[hidenode];//隐含结点状态值
    double x2[outnode];//输出结点状态值
    double o1[hidenode];//隐含层激活值
    double o2[hidenode];//输出层激活值

    for(int isamp=0;isamp<trainsample;isamp++)//循环训练一次样品
    {
        for(int i=0;i<innode;i++)
            x[i]=p[isamp][i]; //输入的样本
        for(int i=0;i<outnode;i++)
            yd[i]=t[isamp][i]; //期望输出的样本

        //构造每个样品的输入和输出标准
        for(int j=0;j<hidenode;j++)
        {
            o1[j]=0.0;
            for(int i=0;i<innode;i++)
                o1[j]=o1[j]+w[i][j]*x[i];//隐含层各单元输入激活值
            x1[j]=1.0/(1+exp(-o1[j]-b1[j]));//隐含层各单元的输出
            //    if(o1[j]+b1[j]>0) x1[j]=1;
            //else x1[j]=0;
        }

        for(int k=0;k<outnode;k++)
        {
            o2[k]=0.0;
            for(int j=0;j<hidenode;j++)
                o2[k]=o2[k]+w1[j][k]*x1[j]; //输出层各单元输入激活值
            x2[k]=1.0/(1.0+exp(-o2[k]-b2[k])); //输出层各单元输出
            //    if(o2[k]+b2[k]>0) x2[k]=1;
            //    else x2[k]=0;
        }

        for(int k=0;k<outnode;k++)
        {
            qq[k]=(yd[k]-x2[k])*x2[k]*(1-x2[k]); //希望输出与实际输出的偏差
            for(int j=0;j<hidenode;j++)
                w1[j][k]+=rate_w1*qq[k]*x1[j];  //下一次的隐含层和输出层之间的新连接权
        }

        for(int j=0;j<hidenode;j++)
        {
            pp[j]=0.0;
            for(int k=0;k<outnode;k++)
                pp[j]=pp[j]+qq[k]*w1[j][k];
            pp[j]=pp[j]*x1[j]*(1-x1[j]); //隐含层的校正误差

            for(int i=0;i<innode;i++)
                w[i][j]+=rate_w*pp[j]*x[i]; //下一次的输入层和隐含层之间的新连接权
        }

        for(int k=0;k<outnode;k++)
        {
            e+=fabs(yd[k]-x2[k])*fabs(yd[k]-x2[k]); //计算均方差
        }
        error=e/2.0;

        for(int k=0;k<outnode;k++)
            b2[k]=b2[k]+rate_b2*qq[k]; //下一次的隐含层和输出层之间的新阈值
        for(int j=0;j<hidenode;j++)
            b1[j]=b1[j]+rate_b1*pp[j]; //下一次的输入层和隐含层之间的新阈值
    }
}

/*BP网络的测试函数，给出对应输入的输出*/
double *BpNet::recognize(double *p)
{
    double x[innode]; //输入向量
    double x1[hidenode]; //隐含结点状态值
    double x2[outnode]; //输出结点状态值
    double o1[hidenode]; //隐含层激活值
    double o2[hidenode]; //输出层激活值

    for(int i=0;i<innode;i++)
        x[i]=p[i];

    for(int j=0;j<hidenode;j++)
    {
        o1[j]=0.0;
        for(int i=0;i<innode;i++)
            o1[j]=o1[j]+w[i][j]*x[i]; //隐含层各单元激活值
        x1[j]=1.0/(1.0+exp(-o1[j]-b1[j])); //隐含层各单元输出
        //if(o1[j]+b1[j]>0) x1[j]=1;
        //    else x1[j]=0;
    }

    for(int k=0;k<outnode;k++)
    {
        o2[k]=0.0;
        for(int j=0;j<hidenode;j++)
            o2[k]=o2[k]+w1[j][k]*x1[j];//输出层各单元激活值
        x2[k]=1.0/(1.0+exp(-o2[k]-b2[k]));//输出层各单元输出
        //if(o2[k]+b2[k]>0) x2[k]=1;
        //else x2[k]=0;
    }

    for(int k=0;k<outnode;k++)
    {
        result[k]=x2[k];
    }
    return result;
}

/*保存已经训练好的网络参数的方法*/
void BpNet::writetrain()
{
    FILE *stream0;
    FILE *stream1;
    FILE *stream2;
    FILE *stream3;
    int i,j;
    //隐含结点权值写入
    if(( stream0 = fopen("w1.txt", "w+" ))==NULL)
    {
        cout<<"创建文件失败!";
        exit(1);
    }
    for(i=0;i<innode;i++)
    {
        for(j=0;j<hidenode;j++)
        {
            fprintf(stream0, "%f\n", w[i][j]);
        }
    }
    fclose(stream0);

    //输出结点权值写入
    if(( stream1 = fopen("w2.txt", "w+" ))==NULL)
    {
        cout<<"创建文件失败!";
        exit(1);
    }
    for(i=0;i<hidenode;i++)
    {
        for(j=0;j<outnode;j++)
        {
            fprintf(stream1, "%f\n",w1[i][j]);
        }
    }
    fclose(stream1);

    //隐含结点阀值写入
    if(( stream2 = fopen("b1.txt", "w+" ))==NULL)
    {
        cout<<"创建文件失败!";
        exit(1);
    }
    for(i=0;i<hidenode;i++)
        fprintf(stream2, "%f\n",b1[i]);
    fclose(stream2);

    //输出结点阀值写入
    if(( stream3 = fopen("b2.txt", "w+" ))==NULL)
    {
        cout<<"创建文件失败!";
        exit(1);
    }
    for(i=0;i<outnode;i++)
        fprintf(stream3, "%f\n",b2[i]);
    fclose(stream3);

}

/*读取已经训练好的网络参数*/
void BpNet::readtrain()
{
    FILE *stream0;
    FILE *stream1;
    FILE *stream2;
    FILE *stream3;
    int i,j;

    //隐含结点权值读出
    if(( stream0 = fopen("w.txt", "r" ))==NULL)
    {
        cout<<"打开文件失败!";
        exit(1);
    }
    float  wx[innode][hidenode];
    for(i=0;i<innode;i++)
    {
        for(j=0;j<hidenode;j++)
        {
            fscanf(stream0, "%f", &wx[i][j]);
            w[i][j]=wx[i][j];
        }
    }
    fclose(stream0);

    //输出结点权值读出
    if(( stream1 = fopen("w1.txt", "r" ))==NULL)
    {
        cout<<"打开文件失败!";
        exit(1);
    }
    float  wx1[hidenode][outnode];
    for(i=0;i<hidenode;i++)
    {
        for(j=0;j<outnode;j++)
        {
            fscanf(stream1, "%f", &wx1[i][j]);
            w1[i][j]=wx1[i][j];
        }
    }
    fclose(stream1);

    //隐含结点阀值读出
    if(( stream2 = fopen("b1.txt", "r" ))==NULL)
    {
        cout<<"打开文件失败!";
        exit(1);
    }
    float xb1[hidenode];
    for(i=0;i<hidenode;i++)
    {
        fscanf(stream2, "%f",&xb1[i]);
        b1[i]=xb1[i];
    }
    fclose(stream2);

    //输出结点阀值读出
    if(( stream3 = fopen("b2.txt", "r" ))==NULL)
    {
        cout<<"打开文件失败!";
        exit(1);
    }
    float xb2[outnode];
    for(i=0;i<outnode;i++)
    {
        fscanf(stream3, "%f",&xb2[i]);
        b2[i]=xb2[i];
    }
    fclose(stream3);
}


//输入样本
double X[trainsample][innode]= {
    0
    };
//期望输出样本
double Y[trainsample][outnode]={
    0
    };

/*求出正向最大值所在列，然后将整个数组归一化*/
int compet(double yout[outnode])
{
	double maxvalue = -1;
	int maxindex = 0;
	for( int i = 0; i < outnode; i++ )
		if( maxvalue < yout[i] )
		{
			maxvalue = yout[i];
			maxindex = i;
		}
	memset(yout, 0, sizeof(double)*outnode);
	yout[maxindex] = 1;
	return maxindex;
}

/*找出输出数组中1的位置，用于判断实际输出*/
int find_1(double y[outnode])
{
	for( int i = 0; i < outnode; i++ )
		if( fabs(1- y[i]) < 0.1 )
			return i;
	return -1;
}


unsigned char *pBmpBuf;//读入图像数据的指针
int bmpWidth;//图像的宽
int bmpHeight;//图像的高
RGBQUAD *pColorTable;//颜色表指针
int biBitCount;//图像类型，每像素位数
byte pixeldata[598][722];
int lineByte = 0;
/***********************************************************************
*函数名称：readBmp()
*函数参数：char *bmpName -文件名字及路径
*返回值：0为失败，1为成功
*说明： 给定一个图像文件名及其路径，读图像的位图数据、宽、高、颜色表及每像素
        *位数等数据进内存，存放在相应的全局变量中
***********************************************************************/
bool readBmp(FILE *fp)
{

    //跳过位图文件头结构BITMAPFILEHEADER
    fseek(fp, sizeof(BITMAPFILEHEADER),0);
    //定义位图信息头结构变量，读取位图信息头进内存，存放在变量head中
    BITMAPINFOHEADER head;
    fread(&head, sizeof(BITMAPINFOHEADER), 1,fp); 
    //获取图像宽、高、每像素所占位数等信息
    bmpWidth = head.biWidth;
    bmpHeight = head.biHeight;
    biBitCount = head.biBitCount;
    //定义变量，计算图像每行像素所占的字节数（必须是4的倍数）
    lineByte=(bmpWidth * biBitCount/8+3)/4*4;
    //灰度图像有颜色表，且颜色表表项为256
    if(biBitCount==8)   //申请颜色表所需要的空间，读颜色表进内存
    {
        pColorTable=new RGBQUAD[256];
        fread(pColorTable,sizeof(RGBQUAD),256,fp);
    }
    //申请位图数据所需要的空间，读位图数据进内存
    pBmpBuf=new unsigned char[lineByte * bmpHeight];

    fread(pBmpBuf,1,lineByte * bmpHeight,fp);
    //关闭文件
    fclose(fp);
    return 1;
}

/*本函数处理已经读入的图像数据，生成测试数据和学习数据两个文件*/
void data_presovle(FILE *datafile)
{
	readBmp(datafile);
	for( int i = 0; i < bmpHeight; i++ )
	{
		for( int j = 0; j < bmpWidth; j++)
		{
			int k = bmpHeight - i-1;
			pixeldata[k][j] = pBmpBuf[i*lineByte+j];// 转存图像的像素矩阵
		}
	}
	FILE *trainfile = fopen("traindata.txt","w");
	FILE *testfile = fopen("testdata.txt","w");
	for(int i = 0; i < 26; i++ )
		for(int j = 0; j < 38; j+= 2 )
		{
			for( int r = 0; r < 23; r++ )
			{
				for( int c = 0; c < 19; c++ )
				{
					fprintf(trainfile, "%d ",pixeldata[i*23+r][j*19+c] > 127 ? 1: 0);
					fprintf(testfile, "%d ",pixeldata[i*23+r][(j+1)*19+c] > 127 ? 1: 0);
				}
			}

			for( int k = 0; k < 26; k++ )
			{
				fprintf(trainfile,"%d ", k==i?1:0);
				fprintf(testfile,"%d ",k == i ? 1:0);
			}
		}
		fclose(trainfile);
		fclose(testfile);
		
}
int main()
{
    BpNet bp;
	
	srand((int)time(0));

    bp.init();
	FILE *bmpfile = fopen("photo.bmp","rb");
	if( bmpfile == NULL )
	{
		printf("无法打开photo.bmp!\n");
		system("pause");
		return -1;
	}
	data_presovle(bmpfile);

    int times=0;
	FILE *traindata = fopen("traindata.txt","r");
	if( traindata == NULL )
	{
		printf("无法读取训练文件!\n");
		system("pause");
		return -2;
	}
	FILE *errorfile = fopen("error.txt","w");
	for( int k = 0; k < trainsample; k++ )
	{
		int temp = 0;
		for( int i = 0; i < innode; i++ )
		{
			fscanf(traindata, "%d", &temp);
			X[k][i] = temp;
		}
		for( int i = 0; i < outnode; i++ )
		{
			fscanf(traindata, "%d", &temp);
			Y[k][i] = temp;
		}
	}
	printf("开始训练网络，参数如下：\n");
	printf("输入节点个数：%d\n输出节点个数：%d\n隐含节点个数：%d\n", innode, outnode, hidenode);
	printf("学习速率：%.2f", bp.rate_b1);
	printf("均方误差目标：<= %f\n",target_error);
	printf("每20次显示一次训练结果\n");
    while(bp.error > target_error)
    {
        bp.e=0.0;

        bp.train(X,Y);
		if( times%20 == 0 )
		{
		  printf("训练次数：%d  当前误差：%lf \n",times, bp.error);
		  fprintf(errorfile, "%f ",bp.error);
		}
		times++;
        
    }
	printf("训练完成！共训练%d次，最终误差%lf\n",times, bp.error);
	fprintf(errorfile, "%f ",bp.error);
	fclose(errorfile);
	bp.writetrain();
	FILE *testfile = fopen("testdata.txt","r");
	if( testfile == NULL )
	{
		printf("无法读取测试文件！\n");
		system("pause");
		return -2;
	}

    double testin[innode]={0};
	double testout[outnode] = {0};
	int right = 0;
	int sum = 190;

	for( int i = 0; i < 190; i++ )
	{
		int temp = 0;
		for( int j = 0; j < innode; j++ )
		{
			fscanf(testfile,"%d",&temp);
			testin[j] = temp;
		}
		for( int j = 0; j < outnode; j++ )
		{
			fscanf(testfile,"%d",&temp);
			testout[j] = temp;
		}
		double *re = bp.recognize(testin);
		if( compet(re) == find_1(testout) )
		{
			right++;
			printf("识别正确！正确结果：%d \n",find_1(testout));
		}
		else printf("识别错误！正确识别:%d 神经网络识别: %d \n",find_1(testout), compet(re));
	}

	printf("\n\n共190组测试样本，其中识别正确 %d 组，正确率%d%%\n",right, right*100/190);
   system("pause");

	
    return 0;
}