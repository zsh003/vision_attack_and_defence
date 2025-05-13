#include "AllHeader.h"


// ��ص͵�ѹ��������ֵ����50��������ӳ�ʱ�䣬��λΪ���롣
// ���磺50*20=1000����1�롣
#define Junce_CHECK_COUNT        20
int Voltage_JUNCE = 0;        // ��ص�ѹֵ
int Jucet_Low_Count = 0;   // �������ؼ���
u8 J_bat_state = 1;

void J_Adc_init(void)
{
	ADC_InitTypeDef ADC_InitStructure; 
	GPIO_InitTypeDef GPIO_InitStructure;

	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOC |RCC_APB2Periph_ADC1	, ENABLE );	  //ʹ��ADC1ͨ��ʱ��
 

	RCC_ADCCLKConfig(RCC_PCLK2_Div6);   //����ADC��Ƶ����6 72M/6=12,ADC���ʱ�䲻�ܳ���14M

	//PC1 ��Ϊģ��ͨ����������                         
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_1;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;		//ģ����������
	GPIO_Init(GPIOC, &GPIO_InitStructure);	

	ADC_DeInit(ADC1);  //��λADC1 

	ADC_InitStructure.ADC_Mode = ADC_Mode_Independent;	//ADC����ģʽ:ADC1��ADC2�����ڶ���ģʽ
	ADC_InitStructure.ADC_ScanConvMode = DISABLE;	//ģ��ת�������ڵ�ͨ��ģʽ
	ADC_InitStructure.ADC_ContinuousConvMode = DISABLE;	//ģ��ת�������ڵ���ת��ģʽ
	ADC_InitStructure.ADC_ExternalTrigConv = ADC_ExternalTrigConv_None;	//ת��������������ⲿ��������
	ADC_InitStructure.ADC_DataAlign = ADC_DataAlign_Right;	//ADC�����Ҷ���
	ADC_InitStructure.ADC_NbrOfChannel = 1;	//˳����й���ת����ADCͨ������Ŀ
	ADC_Init(ADC1, &ADC_InitStructure);	//����ADC_InitStruct��ָ���Ĳ�����ʼ������ADCx�ļĴ���   

  
	ADC_Cmd(ADC1, ENABLE);	//ʹ��ָ����ADC1
	
	ADC_ResetCalibration(ADC1);	//ʹ�ܸ�λУ׼  
	 
	while(ADC_GetResetCalibrationStatus(ADC1));	//�ȴ���λУ׼����
	
	ADC_StartCalibration(ADC1);	 //����ADУ׼
 
	while(ADC_GetCalibrationStatus(ADC1));	 //�ȴ�У׼����
 
//	ADC_SoftwareStartConvCmd(ADC1, ENABLE);		//ʹ��ָ����ADC1�����ת����������

}			

//���ADCֵ
//ch:ͨ��ֵ 0~3
u16 J_Get_Adc(u8 ch)   
{
  	//����ָ��ADC�Ĺ�����ͨ����һ�����У�����ʱ��
	ADC_RegularChannelConfig(ADC1, ch, 1, ADC_SampleTime_239Cycles5 );	//ADC1,ADCͨ��,����ʱ��Ϊ239.5����	  			    
  
	ADC_SoftwareStartConvCmd(ADC1, ENABLE);		//ʹ��ָ����ADC1�����ת����������	
	 
	while(!ADC_GetFlagStatus(ADC1, ADC_FLAG_EOC ));//�ȴ�ת������

	return ADC_GetConversionValue(ADC1);	//�������һ��ADC1�������ת�����
}

// ��� ADC ��β���ƽ��ֵ, ch:ͨ��ֵ ; times:��������
u16 J_Adc_Get_Average(u8 ch, u8 times)
{
	u32 temp_val = 0;
	u8 t;
	for (t = 0; t < times; t++)
	{
		temp_val += J_Get_Adc(ch);
		// delay_ms(5);
	}
	return temp_val / times;
}

// ��ò��ԭʼ��ѹֵ
float J_Adc_Get_Measure_Volotage(void)
{
	u16 adcx;
	float temp;
	adcx = J_Adc_Get_Average(11, 1);	//ADC Channel 1  ADC Channel 11  
	temp = (float)adcx * (33.0f / 4096);
	return temp;
}

// ���ʵ�ʵ�ط�ѹǰ��ѹ
float J_Adc_Get_Battery_Volotage(void)
{
	float temp;
	temp = J_Adc_Get_Measure_Volotage();
	return temp;
}


// ���أ�1  ��������0
u8 J_Bsp_Bat_State(void)
{

	Voltage_JUNCE = (int) (J_Adc_Get_Battery_Volotage() * 100);
	if (Voltage_JUNCE > 1600)
	{
		Jucet_Low_Count++;
		if(Jucet_Low_Count > Junce_CHECK_COUNT)
		{
			return 1;

		}
	}
	else
	{
		Jucet_Low_Count = 0;
	}
	return 0;
}

//���ص�ǰ�����ĵ�ѹ
int J_Bsp_Bat_Voltage_Z100(void)
{
	return Voltage_JUNCE;
}


// ����ϵͳ�Ƿ���빩����������������1������������0
u8 J_Bsp_System_Enable(void)
{
	return J_bat_state;
}


