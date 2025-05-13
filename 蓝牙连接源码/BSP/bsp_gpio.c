/**
* @par Copyright (C): 2010-2019, Shenzhen Yahboom Tech
* @file         bsp_gpio.c      
* @version      V1.0       
* @brief        ����gpioԴ�ļ�
* @details      
* @par History  ������˵��
*                 
* version:	
*/

#include "AllHeader.h"


/**
* Function       Servo_GPIO_Init
* @author        
* @date            
* @brief         ��Ҫ�õ��Ķ����ʼ���ӿ�
* @param[in]     void
* @param[out]    void
* @retval        void
* @par History   ��
*/

void Servo_GPIO_Init(void)
{		
	/*����һ��GPIO_InitTypeDef���͵Ľṹ��*/
	GPIO_InitTypeDef GPIO_InitStructure;

	PWR_BackupAccessCmd(ENABLE);//�����޸�RTC �ͺ󱸼Ĵ���
	RCC_LSEConfig(RCC_LSE_OFF);//�ر��ⲿ�����ⲿʱ���źŹ��� ��PC13 PC14 PC15 �ſ��Ե���ͨIO�á�
	BKP_TamperPinCmd(DISABLE);//�ر����ּ�⹦�ܣ�Ҳ���� PC13��Ҳ���Ե���ͨIO ʹ��
	PWR_BackupAccessCmd(DISABLE);//��ֹ�޸ĺ󱸼Ĵ���
	
	
	
#ifdef USE_SERVO_J1
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J1_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J1_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J1_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J2
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J2_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J2_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz; 
	/*���ÿ⺯������ʼ��Servo_J2_PORT*/
  	GPIO_Init(Servo_J2_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J3
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J3_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J3_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz; 
	/*���ÿ⺯������ʼ��Servo_J3_PORT*/
  	GPIO_Init(Servo_J3_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J4
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J4_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J4_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J4_PORT*/
  	GPIO_Init(Servo_J4_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J5
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J5_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J5_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J5_PORT*/
  	GPIO_Init(Servo_J5_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J6
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J6_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J6_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J6_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J7
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J7_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J7_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J7_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J8
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J8_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J8_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J8_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J9
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J9_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J9_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J9_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J10
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J10_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J10_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J10_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J11
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J11_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J11_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J11_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J12
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J12_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J12_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J12_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J13
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J13_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J13_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J13_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J14
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J14_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J14_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J14_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J15
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J15_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J15_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J15_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J16
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J16_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J16_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J16_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J17
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J17_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J17_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J17_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J18
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J18_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J18_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J18_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J19
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J19_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J19_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J19_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J20
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J20_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J20_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J20_PORT, &GPIO_InitStructure);		  
#endif
 
#ifdef USE_SERVO_J21
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J21_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J21_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J21_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J22
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J22_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J22_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J22_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J23
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J23_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J23_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J23_PORT, &GPIO_InitStructure);		  
#endif

#ifdef USE_SERVO_J24
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(Servo_J24_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = Servo_J24_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��Servo_J1_PORT*/
  	GPIO_Init(Servo_J24_PORT, &GPIO_InitStructure);		  
#endif
}

