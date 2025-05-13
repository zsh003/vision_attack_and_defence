#include "AllHeader.h"

/*********************************************************     
**********************************************************/	 
//ʹ�õĲ���Ӳ��SPI
#define DELAY_TIME  delay_us(5); 

u16 Handkey;	// ����ֵ��ȡ����ʱ�洢��
u8 Comd[2]={0x01,0x42};	//��ʼ�����������
u8 Data[9]={0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}; //���ݴ洢����
u16 MASK[]={
    PSB_SELECT,
    PSB_L3,
    PSB_R3 ,
    PSB_START,
    PSB_PAD_UP,
    PSB_PAD_RIGHT,
    PSB_PAD_DOWN,
    PSB_PAD_LEFT,
    PSB_L2,
    PSB_R2,
    PSB_L1,
    PSB_R1 ,
    PSB_GREEN,
    PSB_RED,
    PSB_BLUE,
    PSB_PINK
	};	
 /**************************************************************************
�������ܣ�������PS2������ģ��ĳ�ʼ������
��ڲ�������
����  ֵ����
����ֵ�밴����
�ֱ��ӿڳ�ʼ��    ����  DI->PA6 
                 ���  DO->PA7    CS->PA4  CLK->PA5
**************************************************************************/
void PS2_Init(void)
{
	GPIO_InitTypeDef GPIO_InitStructure;                           //����GPIO_InitStructure�ṹ��
	//DI
	RCC_APB2PeriphClockCmd(PS_RCC_DI,ENABLE);
	GPIO_InitStructure.GPIO_Mode=GPIO_Mode_IPD;                     //��������ģʽ
	GPIO_InitStructure.GPIO_Pin=PS_PIN_DI;													
	GPIO_InitStructure.GPIO_Speed=GPIO_Speed_50MHz;                //50MHZ  
	GPIO_Init(PS_PORT_DI,&GPIO_InitStructure);                          
	
	//DO
	RCC_APB2PeriphClockCmd(PS_RCC_DO,ENABLE);           //����GPIOBʱ��
	GPIO_InitStructure.GPIO_Mode=GPIO_Mode_Out_PP;                 //�����������ģʽ
	GPIO_InitStructure.GPIO_Pin=PS_PIN_DO;  //DO�� 
	GPIO_InitStructure.GPIO_Speed=GPIO_Speed_50MHz;             	 //50MHZ
	GPIO_Init(PS_PORT_DO,&GPIO_InitStructure);                         

	//CS
	RCC_APB2PeriphClockCmd(PS_RCC_CS,ENABLE);           //����GPIOBʱ��
	GPIO_InitStructure.GPIO_Mode=GPIO_Mode_Out_PP;                 //�����������ģʽ
	GPIO_InitStructure.GPIO_Pin=PS_PIN_CS;  //CS�� 
	GPIO_InitStructure.GPIO_Speed=GPIO_Speed_50MHz;             	 //50MHZ
	GPIO_Init(PS_PORT_CS,&GPIO_InitStructure);   
	
	//SCK
	RCC_APB2PeriphClockCmd(PS_RCC_CLK,ENABLE);           //����GPIOBʱ��
	GPIO_InitStructure.GPIO_Mode=GPIO_Mode_Out_PP;                 //�����������ģʽ
	GPIO_InitStructure.GPIO_Pin=PS_PIN_CLK;  //CLK�� 
	GPIO_InitStructure.GPIO_Speed=GPIO_Speed_50MHz;             	 //50MHZ
	GPIO_Init(PS_PORT_CLK,&GPIO_InitStructure);   
	
	
}
/**************************************************************************
�������ܣ����ֱ���������
��ڲ�����CMDָ��
����  ֵ����
**************************************************************************/
void PS2_Cmd(u8 CMD)
{
	volatile u16 ref=0x01;
	Data[1] = 0;
	for(ref=0x01;ref<0x0100;ref<<=1)
	{
		if(ref&CMD)
		{
			DO_H;                   //���һλ����λ
		}
		else DO_L;
		CLK_H;                        //ʱ������
		DELAY_TIME;
		CLK_L;
		DELAY_TIME;
		CLK_H;
		if(DI) //Ϊ�ߵ�ƽ��ʱ��
			Data[1] = ref|Data[1];
	}
	delay_us(16);
}
/**************************************************************************
�������ܣ��ж��Ƿ�Ϊ���ģʽ,0x41=ģ���̵ƣ�0x73=ģ����
��ڲ�����CMDָ��
����  ֵ��0�����ģʽ  ����������ģʽ
**************************************************************************/
u8 PS2_RedLight(void)
{
	CS_L;
	PS2_Cmd(Comd[0]);  //��ʼ����
	PS2_Cmd(Comd[1]);  //��������
	CS_H;
	if( Data[1] == 0X73)   return 0 ;
	else return 1;
}
/**************************************************************************
�������ܣ���ȡ�ֱ�����
��ڲ�������
����  ֵ����
**************************************************************************/
void PS2_ReadData(void)
{
	volatile u8 byte=0;
	volatile u16 ref=0x01;
	CS_L;
	PS2_Cmd(Comd[0]);  //��ʼ����
	PS2_Cmd(Comd[1]);  //��������
	for(byte=2;byte<9;byte++)          //��ʼ��������
	{
		for(ref=0x01;ref<0x100;ref<<=1)
		{
			CLK_H;
			DELAY_TIME;
			CLK_L;
			DELAY_TIME;
			CLK_H;
		      if(DI)
		      Data[byte] = ref|Data[byte];
		}
        delay_us(16);
	}
	CS_H;
}
/**************************************************************************
�������ܣ��Զ�������PS2�����ݽ��д���,ֻ����������
��ڲ�����CMDָ��
����  ֵ����  
//ֻ��һ����������ʱ����Ϊ0�� δ����Ϊ1
**************************************************************************/
u8 PS2_DataKey()
{
	u8 index;
	PS2_ClearData();
	PS2_ReadData();
	Handkey=(Data[4]<<8)|Data[3];     //����16������  ����Ϊ0�� δ����Ϊ1
	for(index=0;index<16;index++)
	{	    
		if((Handkey&(1<<(MASK[index]-1)))==0)
		return index+1;
	}
	return 0;          //û���κΰ�������
}
/**************************************************************************
�������ܣ����ֱ���������
��ڲ������õ�һ��ҡ�˵�ģ����	 ��Χ0~256
����  ֵ����
**************************************************************************/
u8 PS2_AnologData(u8 button)
{
	return Data[button];
}
//������ݻ�����
void PS2_ClearData()
{
	u8 a;
	for(a=0;a<9;a++)
		Data[a]=0x00;
}
/******************************************************
��������: �ֱ��𶯺�����
Calls:		 void PS2_Cmd(u8 CMD);
��ڲ���: motor1:�Ҳ�С�𶯵�� 0x00�أ�������
	        motor2:�����𶯵�� 0x40~0xFF �������ֵԽ�� ��Խ��
����  ֵ:��
******************************************************/
void PS2_Vibration(u8 motor1, u8 motor2)
{
	CS_L;
	delay_us(16);
    PS2_Cmd(0x01);  //��ʼ����
	PS2_Cmd(0x42);  //��������
	PS2_Cmd(0X00);
	PS2_Cmd(motor1);
	PS2_Cmd(motor2);
	PS2_Cmd(0X00);
	PS2_Cmd(0X00);
	PS2_Cmd(0X00);
	PS2_Cmd(0X00);
	CS_H;
	delay_us(16);  
}
/**************************************************************************
�������ܣ�short poll
��ڲ�������
����  ֵ����
**************************************************************************/
void PS2_ShortPoll(void)
{
	CS_L;
	delay_us(16);
	PS2_Cmd(0x01);  
	PS2_Cmd(0x42);  
	PS2_Cmd(0X00);
	PS2_Cmd(0x00);
	PS2_Cmd(0x00);
	CS_H;
	delay_us(16);	
}
/**************************************************************************
�������ܣ���������
��ڲ�������
����  ֵ����
**************************************************************************/
void PS2_EnterConfing(void)
{
    CS_L;
	delay_us(16);
	PS2_Cmd(0x01);  
	PS2_Cmd(0x43);  
	PS2_Cmd(0X00);
	PS2_Cmd(0x01);
	PS2_Cmd(0x00);
	PS2_Cmd(0X00);
	PS2_Cmd(0X00);
	PS2_Cmd(0X00);
	PS2_Cmd(0X00);
	CS_H;
	delay_us(16);
}
/**************************************************************************
�������ܣ�����ģʽ����
��ڲ�������
����  ֵ����
**************************************************************************/
void PS2_TurnOnAnalogMode(void)
{
	CS_L;
	PS2_Cmd(0x01);  
	PS2_Cmd(0x44);  
	PS2_Cmd(0X00);
	PS2_Cmd(0x01); //analog=0x01;digital=0x00  ������÷���ģʽ
	PS2_Cmd(0x03); //Ox03�������ã�������ͨ��������MODE������ģʽ��
				   //0xEE������������ã���ͨ��������MODE������ģʽ��
	PS2_Cmd(0X00);
	PS2_Cmd(0X00);
	PS2_Cmd(0X00);
	PS2_Cmd(0X00);
	CS_H;
	delay_us(16);
}
/**************************************************************************
�������ܣ�������
��ڲ�������
����  ֵ����
**************************************************************************/
void PS2_VibrationMode(void)
{
	CS_L;
	delay_us(16);
	PS2_Cmd(0x01);  
	PS2_Cmd(0x4D);  
	PS2_Cmd(0X00);
	PS2_Cmd(0x00);
	PS2_Cmd(0X01);
	CS_H;
	delay_us(16);	
}
/**************************************************************************
�������ܣ���ɲ���������
��ڲ�������
����  ֵ����
**************************************************************************/
void PS2_ExitConfing(void)
{
    CS_L;
	delay_us(16);
	PS2_Cmd(0x01);  
	PS2_Cmd(0x43);  
	PS2_Cmd(0X00);
	PS2_Cmd(0x00);
	PS2_Cmd(0x5A);
	PS2_Cmd(0x5A);
	PS2_Cmd(0x5A);
	PS2_Cmd(0x5A);
	PS2_Cmd(0x5A);
	CS_H;
	delay_us(16);
}
/**************************************************************************
�������ܣ��ֱ����ó�ʼ��
��ڲ�������
����  ֵ����
**************************************************************************/
void PS2_SetInit(void)
{
	PS2_ShortPoll();
	PS2_ShortPoll();
	PS2_ShortPoll();
	PS2_EnterConfing();		//��������ģʽ
	PS2_TurnOnAnalogMode();	//�����̵ơ�����ģʽ����ѡ���Ƿ񱣴�
	//PS2_VibrationMode();	//������ģʽ
	PS2_ExitConfing();		//��ɲ���������
}


