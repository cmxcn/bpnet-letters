
#include <stdafx.h>
#include<iostream>
#include<cmath>
#include<time.h>
#include<Windows.h>

using namespace std;

#define  innode 437  //��������
#define  hidenode 29//���������
#define  outnode 26 //��������
#define  trainsample 190//BPѵ��������
#define  testsample 190 //����������
#define  digit_width 19 //ÿ���ַ����
#define  digit_height 23 //ÿ���ַ��߶�
#define  target_error  0.1 //���ָ��
class BpNet
{
public:
    void train(double p[trainsample][innode ],double t[trainsample][outnode]);//Bpѵ��
    double p[trainsample][innode];     //ѵ������
    double t[trainsample][outnode];    //ѵ�����

    double *recognize(double *p);//Bpʶ��

    void writetrain(); //����ѵ���õ�Ȩֵ
    void readtrain(); //��ѵ���õ�Ȩֵ����ʹ�Ĳ���ÿ��ȥѵ���ˣ�ֻҪ��ѵ����õ�Ȩֵ��������OK

    BpNet();
    virtual ~BpNet();

public:
    void init();
    double w[innode][hidenode];//�������Ȩֵ
    double w1[hidenode][outnode];//������Ȩֵ
    double b1[hidenode];//������㷧ֵ
    double b2[outnode];//�����㷧ֵ

    double rate_w; //Ȩֵѧϰ�ʣ������-������)
    double rate_w1;//Ȩֵѧϰ�� (������-�����)
    double rate_b1;//�����㷧ֵѧϰ��
    double rate_b2;//����㷧ֵѧϰ��

    double e;//������
    double error;//�����������
    double result[outnode];// Bp���
};

BpNet::BpNet()
{
    error=1.0;
    e=0.0;

    rate_w=0.95;  //Ȩֵѧϰ�ʣ������--������)
    rate_w1=0.95; //Ȩֵѧϰ�� (������--�����)
    rate_b1=0.95; //�����㷧ֵѧϰ��
    rate_b2=0.95; //����㷧ֵѧϰ��
}

BpNet::~BpNet()
{

}

void winit(double w[],int n) //Ȩֵ��ʼ��
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

/*BP����ĵ���ѵ���������������������Ϊlogsig����*/
void BpNet::train(double p[trainsample][innode],double t[trainsample][outnode])
{
    double pp[hidenode];//��������У�����
    double qq[outnode];//ϣ�����ֵ��ʵ�����ֵ��ƫ��
    double yd[outnode];//ϣ�����ֵ

    double x[innode]; //��������
    double x1[hidenode];//�������״ֵ̬
    double x2[outnode];//������״ֵ̬
    double o1[hidenode];//�����㼤��ֵ
    double o2[hidenode];//����㼤��ֵ

    for(int isamp=0;isamp<trainsample;isamp++)//ѭ��ѵ��һ����Ʒ
    {
        for(int i=0;i<innode;i++)
            x[i]=p[isamp][i]; //���������
        for(int i=0;i<outnode;i++)
            yd[i]=t[isamp][i]; //�������������

        //����ÿ����Ʒ������������׼
        for(int j=0;j<hidenode;j++)
        {
            o1[j]=0.0;
            for(int i=0;i<innode;i++)
                o1[j]=o1[j]+w[i][j]*x[i];//���������Ԫ���뼤��ֵ
            x1[j]=1.0/(1+exp(-o1[j]-b1[j]));//���������Ԫ�����
            //    if(o1[j]+b1[j]>0) x1[j]=1;
            //else x1[j]=0;
        }

        for(int k=0;k<outnode;k++)
        {
            o2[k]=0.0;
            for(int j=0;j<hidenode;j++)
                o2[k]=o2[k]+w1[j][k]*x1[j]; //��������Ԫ���뼤��ֵ
            x2[k]=1.0/(1.0+exp(-o2[k]-b2[k])); //��������Ԫ���
            //    if(o2[k]+b2[k]>0) x2[k]=1;
            //    else x2[k]=0;
        }

        for(int k=0;k<outnode;k++)
        {
            qq[k]=(yd[k]-x2[k])*x2[k]*(1-x2[k]); //ϣ�������ʵ�������ƫ��
            for(int j=0;j<hidenode;j++)
                w1[j][k]+=rate_w1*qq[k]*x1[j];  //��һ�ε�������������֮���������Ȩ
        }

        for(int j=0;j<hidenode;j++)
        {
            pp[j]=0.0;
            for(int k=0;k<outnode;k++)
                pp[j]=pp[j]+qq[k]*w1[j][k];
            pp[j]=pp[j]*x1[j]*(1-x1[j]); //�������У�����

            for(int i=0;i<innode;i++)
                w[i][j]+=rate_w*pp[j]*x[i]; //��һ�ε�������������֮���������Ȩ
        }

        for(int k=0;k<outnode;k++)
        {
            e+=fabs(yd[k]-x2[k])*fabs(yd[k]-x2[k]); //���������
        }
        error=e/2.0;

        for(int k=0;k<outnode;k++)
            b2[k]=b2[k]+rate_b2*qq[k]; //��һ�ε�������������֮�������ֵ
        for(int j=0;j<hidenode;j++)
            b1[j]=b1[j]+rate_b1*pp[j]; //��һ�ε�������������֮�������ֵ
    }
}

/*BP����Ĳ��Ժ�����������Ӧ��������*/
double *BpNet::recognize(double *p)
{
    double x[innode]; //��������
    double x1[hidenode]; //�������״ֵ̬
    double x2[outnode]; //������״ֵ̬
    double o1[hidenode]; //�����㼤��ֵ
    double o2[hidenode]; //����㼤��ֵ

    for(int i=0;i<innode;i++)
        x[i]=p[i];

    for(int j=0;j<hidenode;j++)
    {
        o1[j]=0.0;
        for(int i=0;i<innode;i++)
            o1[j]=o1[j]+w[i][j]*x[i]; //���������Ԫ����ֵ
        x1[j]=1.0/(1.0+exp(-o1[j]-b1[j])); //���������Ԫ���
        //if(o1[j]+b1[j]>0) x1[j]=1;
        //    else x1[j]=0;
    }

    for(int k=0;k<outnode;k++)
    {
        o2[k]=0.0;
        for(int j=0;j<hidenode;j++)
            o2[k]=o2[k]+w1[j][k]*x1[j];//��������Ԫ����ֵ
        x2[k]=1.0/(1.0+exp(-o2[k]-b2[k]));//��������Ԫ���
        //if(o2[k]+b2[k]>0) x2[k]=1;
        //else x2[k]=0;
    }

    for(int k=0;k<outnode;k++)
    {
        result[k]=x2[k];
    }
    return result;
}

/*�����Ѿ�ѵ���õ���������ķ���*/
void BpNet::writetrain()
{
    FILE *stream0;
    FILE *stream1;
    FILE *stream2;
    FILE *stream3;
    int i,j;
    //�������Ȩֵд��
    if(( stream0 = fopen("w1.txt", "w+" ))==NULL)
    {
        cout<<"�����ļ�ʧ��!";
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

    //������Ȩֵд��
    if(( stream1 = fopen("w2.txt", "w+" ))==NULL)
    {
        cout<<"�����ļ�ʧ��!";
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

    //������㷧ֵд��
    if(( stream2 = fopen("b1.txt", "w+" ))==NULL)
    {
        cout<<"�����ļ�ʧ��!";
        exit(1);
    }
    for(i=0;i<hidenode;i++)
        fprintf(stream2, "%f\n",b1[i]);
    fclose(stream2);

    //�����㷧ֵд��
    if(( stream3 = fopen("b2.txt", "w+" ))==NULL)
    {
        cout<<"�����ļ�ʧ��!";
        exit(1);
    }
    for(i=0;i<outnode;i++)
        fprintf(stream3, "%f\n",b2[i]);
    fclose(stream3);

}

/*��ȡ�Ѿ�ѵ���õ��������*/
void BpNet::readtrain()
{
    FILE *stream0;
    FILE *stream1;
    FILE *stream2;
    FILE *stream3;
    int i,j;

    //�������Ȩֵ����
    if(( stream0 = fopen("w.txt", "r" ))==NULL)
    {
        cout<<"���ļ�ʧ��!";
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

    //������Ȩֵ����
    if(( stream1 = fopen("w1.txt", "r" ))==NULL)
    {
        cout<<"���ļ�ʧ��!";
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

    //������㷧ֵ����
    if(( stream2 = fopen("b1.txt", "r" ))==NULL)
    {
        cout<<"���ļ�ʧ��!";
        exit(1);
    }
    float xb1[hidenode];
    for(i=0;i<hidenode;i++)
    {
        fscanf(stream2, "%f",&xb1[i]);
        b1[i]=xb1[i];
    }
    fclose(stream2);

    //�����㷧ֵ����
    if(( stream3 = fopen("b2.txt", "r" ))==NULL)
    {
        cout<<"���ļ�ʧ��!";
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


//��������
double X[trainsample][innode]= {
    0
    };
//�����������
double Y[trainsample][outnode]={
    0
    };

/*����������ֵ�����У�Ȼ�����������һ��*/
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

/*�ҳ����������1��λ�ã������ж�ʵ�����*/
int find_1(double y[outnode])
{
	for( int i = 0; i < outnode; i++ )
		if( fabs(1- y[i]) < 0.1 )
			return i;
	return -1;
}


unsigned char *pBmpBuf;//����ͼ�����ݵ�ָ��
int bmpWidth;//ͼ��Ŀ�
int bmpHeight;//ͼ��ĸ�
RGBQUAD *pColorTable;//��ɫ��ָ��
int biBitCount;//ͼ�����ͣ�ÿ����λ��
byte pixeldata[598][722];
int lineByte = 0;
/***********************************************************************
*�������ƣ�readBmp()
*����������char *bmpName -�ļ����ּ�·��
*����ֵ��0Ϊʧ�ܣ�1Ϊ�ɹ�
*˵���� ����һ��ͼ���ļ�������·������ͼ���λͼ���ݡ����ߡ���ɫ��ÿ����
        *λ�������ݽ��ڴ棬�������Ӧ��ȫ�ֱ�����
***********************************************************************/
bool readBmp(FILE *fp)
{

    //����λͼ�ļ�ͷ�ṹBITMAPFILEHEADER
    fseek(fp, sizeof(BITMAPFILEHEADER),0);
    //����λͼ��Ϣͷ�ṹ��������ȡλͼ��Ϣͷ���ڴ棬����ڱ���head��
    BITMAPINFOHEADER head;
    fread(&head, sizeof(BITMAPINFOHEADER), 1,fp); 
    //��ȡͼ����ߡ�ÿ������ռλ������Ϣ
    bmpWidth = head.biWidth;
    bmpHeight = head.biHeight;
    biBitCount = head.biBitCount;
    //�������������ͼ��ÿ��������ռ���ֽ�����������4�ı�����
    lineByte=(bmpWidth * biBitCount/8+3)/4*4;
    //�Ҷ�ͼ������ɫ������ɫ�����Ϊ256
    if(biBitCount==8)   //������ɫ������Ҫ�Ŀռ䣬����ɫ����ڴ�
    {
        pColorTable=new RGBQUAD[256];
        fread(pColorTable,sizeof(RGBQUAD),256,fp);
    }
    //����λͼ��������Ҫ�Ŀռ䣬��λͼ���ݽ��ڴ�
    pBmpBuf=new unsigned char[lineByte * bmpHeight];

    fread(pBmpBuf,1,lineByte * bmpHeight,fp);
    //�ر��ļ�
    fclose(fp);
    return 1;
}

/*�����������Ѿ������ͼ�����ݣ����ɲ������ݺ�ѧϰ���������ļ�*/
void data_presovle(FILE *datafile)
{
	readBmp(datafile);
	for( int i = 0; i < bmpHeight; i++ )
	{
		for( int j = 0; j < bmpWidth; j++)
		{
			int k = bmpHeight - i-1;
			pixeldata[k][j] = pBmpBuf[i*lineByte+j];// ת��ͼ������ؾ���
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
		printf("�޷���photo.bmp!\n");
		system("pause");
		return -1;
	}
	data_presovle(bmpfile);

    int times=0;
	FILE *traindata = fopen("traindata.txt","r");
	if( traindata == NULL )
	{
		printf("�޷���ȡѵ���ļ�!\n");
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
	printf("��ʼѵ�����磬�������£�\n");
	printf("����ڵ������%d\n����ڵ������%d\n�����ڵ������%d\n", innode, outnode, hidenode);
	printf("ѧϰ���ʣ�%.2f", bp.rate_b1);
	printf("�������Ŀ�꣺<= %f\n",target_error);
	printf("ÿ20����ʾһ��ѵ�����\n");
    while(bp.error > target_error)
    {
        bp.e=0.0;

        bp.train(X,Y);
		if( times%20 == 0 )
		{
		  printf("ѵ��������%d  ��ǰ��%lf \n",times, bp.error);
		  fprintf(errorfile, "%f ",bp.error);
		}
		times++;
        
    }
	printf("ѵ����ɣ���ѵ��%d�Σ��������%lf\n",times, bp.error);
	fprintf(errorfile, "%f ",bp.error);
	fclose(errorfile);
	bp.writetrain();
	FILE *testfile = fopen("testdata.txt","r");
	if( testfile == NULL )
	{
		printf("�޷���ȡ�����ļ���\n");
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
			printf("ʶ����ȷ����ȷ�����%d \n",find_1(testout));
		}
		else printf("ʶ�������ȷʶ��:%d ������ʶ��: %d \n",find_1(testout), compet(re));
	}

	printf("\n\n��190���������������ʶ����ȷ %d �飬��ȷ��%d%%\n",right, right*100/190);
   system("pause");

	
    return 0;
}