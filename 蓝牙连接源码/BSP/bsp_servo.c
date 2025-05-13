/**
* @par Copyright (C): 2010-2019, Shenzhen Yahboom Tech
* @file         bsp_servo.c	
* @version      V1.0
* @brief        6�������������Դ�ļ�
* @details      
* @par History  ������˵��
*                 
* 
*/

#include "bsp_servo.h"
#include "delay.h"

//�Ż�������ɣ�(2ά����)
/*�Ƕȷ�Χ 0~180*/
int Angle_J[GROUP_NUM][DUOJI_NUM];


/**
* Function       Servo_J1
* @author        
* @date          
* @brief         ���1���ƺ���
* @param[in]     v_iAngle �Ƕȣ�0~180��
* @param[out]    void
* @retval        void
* @par History   ��
*/
void Servo_J1(int v_iAngle)/*����һ�����庯��������ģ�ⷽʽ����PWMֵ*/
{
	int pulsewidth;    						//�����������

	pulsewidth = (v_iAngle * 11) + 500;			//���Ƕ�ת��Ϊ500-2480 ������ֵ

	GPIO_SetBits(Servo_J1_PORT, Servo_J1_PIN );		//������ӿڵ�ƽ�ø�
	delay_us(pulsewidth);					//��ʱ����ֵ��΢����

	GPIO_ResetBits(Servo_J1_PORT, Servo_J1_PIN );	//������ӿڵ�ƽ�õ�
	delay_ms(20 - pulsewidth/1000);			//��ʱ������ʣ��ʱ��
}

/**
* Function       Servo_J2
* @author        
* @date            
* @brief         ���2���ƺ���
* @param[in]     v_iAngle �Ƕȣ�0~180��
* @param[out]    void
* @retval        void
* @par History   ��
*/
void Servo_J2(int v_iAngle)/*����һ�����庯��������ģ�ⷽʽ����PWMֵ*/
{
	int pulsewidth;    						//�����������

	pulsewidth = (v_iAngle * 11) + 500;			//���Ƕ�ת��Ϊ500-2480 ������ֵ

	GPIO_SetBits(Servo_J2_PORT, Servo_J2_PIN );		//������ӿڵ�ƽ�ø�
	delay_us(pulsewidth);					//��ʱ����ֵ��΢����

	GPIO_ResetBits(Servo_J2_PORT, Servo_J2_PIN );	//������ӿڵ�ƽ�õ�
	delay_ms(20 - pulsewidth/1000);			//��ʱ������ʣ��ʱ��
}

/**
* Function       Servo_J3
* @author        
* @date          
* @brief         ���3���ƺ���
* @param[in]     v_iAngle �Ƕȣ�0~180��
* @param[out]    void
* @retval        void
* @par History   ��
*/
void Servo_J3(int v_iAngle)/*����һ�����庯��������ģ�ⷽʽ����PWMֵ*/
{
	int pulsewidth;    						//�����������

	pulsewidth = (v_iAngle * 11) + 500;			//���Ƕ�ת��Ϊ500-2480 ������ֵ

	GPIO_SetBits(Servo_J3_PORT, Servo_J3_PIN );		//������ӿڵ�ƽ�ø�
	delay_us(pulsewidth);					//��ʱ����ֵ��΢����

	GPIO_ResetBits(Servo_J3_PORT, Servo_J3_PIN );	//������ӿڵ�ƽ�õ�
	delay_ms(20 - pulsewidth/1000);			//��ʱ������ʣ��ʱ��
}

/**
* Function       Servo_J4
* @author        
* @date              
* @brief         ���4���ƺ���
* @param[in]     v_iAngle �Ƕȣ�0~180��
* @param[out]    void
* @retval        void
* @par History   ��
*/
void Servo_J4(int v_iAngle)/*����һ�����庯��������ģ�ⷽʽ����PWMֵ*/
{
	int pulsewidth;    						//�����������

	pulsewidth = (v_iAngle * 11) + 500;			//���Ƕ�ת��Ϊ500-2480 ������ֵ

	GPIO_SetBits(Servo_J4_PORT, Servo_J4_PIN );		//������ӿڵ�ƽ�ø�
	delay_us(pulsewidth);					//��ʱ����ֵ��΢����

	GPIO_ResetBits(Servo_J4_PORT, Servo_J4_PIN );	//������ӿڵ�ƽ�õ�
	delay_ms(20 - pulsewidth/1000);			//��ʱ������ʣ��ʱ��
}

/**
* Function       Servo_J5
* @author        
* @date             
* @brief         ���5���ƺ���
* @param[in]     v_iAngle �Ƕȣ�0~180��
* @param[out]    void
* @retval        void
* @par History   ��
*/
void Servo_J5(int v_iAngle)/*����һ�����庯��������ģ�ⷽʽ����PWMֵ*/
{
	int pulsewidth;    						//�����������

	pulsewidth = (v_iAngle * 11) + 500;			//���Ƕ�ת��Ϊ500-2480 ������ֵ

	GPIO_SetBits(Servo_J5_PORT, Servo_J5_PIN );		//������ӿڵ�ƽ�ø�
	delay_us(pulsewidth);					//��ʱ����ֵ��΢����

	GPIO_ResetBits(Servo_J5_PORT, Servo_J5_PIN );	//������ӿڵ�ƽ�õ�
	delay_ms(20 - pulsewidth/1000);			//��ʱ������ʣ��ʱ��
}

/**
* Function       Servo_J6
* @author        
* @date             
* @brief         ���6���ƺ���
* @param[in]     v_iAngle �Ƕȣ�0~180��
* @param[out]    void
* @retval        void
* @par History   ��
*/
void Servo_J6(int v_iAngle)/*����һ�����庯��������ģ�ⷽʽ����PWMֵ*/
{
	int pulsewidth;    						//�����������

	pulsewidth = (v_iAngle * 11) + 500;			//���Ƕ�ת��Ϊ500-2480 ������ֵ

	GPIO_SetBits(Servo_J6_PORT, Servo_J6_PIN );		//������ӿڵ�ƽ�ø�
	delay_us(pulsewidth);					//��ʱ����ֵ��΢����

	GPIO_ResetBits(Servo_J6_PORT, Servo_J6_PIN );	//������ӿڵ�ƽ�õ�
	delay_ms(20 - pulsewidth/1000);			//��ʱ������ʣ��ʱ��
}

/**
* Function       Servo_J7
* @author        
* @date             
* @brief         ���7���ƺ���
* @param[in]     v_iAngle �Ƕȣ�0~180��
* @param[out]    void
* @retval        void
* @par History   ��
*/
void Servo_J7(int v_iAngle)/*����һ�����庯��������ģ�ⷽʽ����PWMֵ*/
{
	int pulsewidth;    						//�����������

	pulsewidth = (v_iAngle * 11) + 500;			//���Ƕ�ת��Ϊ500-2480 ������ֵ

	GPIO_SetBits(Servo_J7_PORT, Servo_J7_PIN );		//������ӿڵ�ƽ�ø�
	delay_us(pulsewidth);					//��ʱ����ֵ��΢����

	GPIO_ResetBits(Servo_J7_PORT, Servo_J7_PIN );	//������ӿڵ�ƽ�õ�
	delay_ms(20 - pulsewidth/1000);			//��ʱ������ʣ��ʱ��
}

/**
* Function       Servo_J8
* @author        
* @date             
* @brief         ���8���ƺ���
* @param[in]     v_iAngle �Ƕȣ�0~180��
* @param[out]    void
* @retval        void
* @par History   ��
*/
void Servo_J8(int v_iAngle)/*����һ�����庯��������ģ�ⷽʽ����PWMֵ*/
{
	int pulsewidth;    						//�����������

	pulsewidth = (v_iAngle * 11) + 500;			//���Ƕ�ת��Ϊ500-2480 ������ֵ

	GPIO_SetBits(Servo_J8_PORT, Servo_J8_PIN );		//������ӿڵ�ƽ�ø�
	delay_us(pulsewidth);					//��ʱ����ֵ��΢����

	GPIO_ResetBits(Servo_J8_PORT, Servo_J8_PIN );	//������ӿڵ�ƽ�õ�
	delay_ms(20 - pulsewidth/1000);			//��ʱ������ʣ��ʱ��
}

/**
* Function       front_detection
* @author        
* @date             
* @brief         ��̨�����ǰ
* @param[in]     void
* @param[out]    void
* @retval        void
* @par History   ��
*/
void front_detection()
{
	int i = 0;
  	//�˴�ѭ���������٣�Ϊ������С�������ϰ���ķ�Ӧ�ٶ�
  	for(i=0; i <= 15; i++) 						//����PWM��������Ч��ʱ�Ա�֤��ת����Ӧ�Ƕ�
  	{
    	Servo_J1(90);						//ģ�����PWM
  	}
}

/**
* Function       left_detection
* @author        
* @date             
* @brief         ��̨�������
* @param[in]     void
* @param[out]    void
* @retval        void
* @par History   ��
*/
void left_detection()
{
	int i = 0;
	for(i = 0; i <= 15; i++) 						//����PWM��������Ч��ʱ�Ա�֤��ת����Ӧ�Ƕ�
	{
		Servo_J1(175);					//ģ�����PWM
	}
}

/**
* Function       right_detection
* @author        
* @date             
* @brief         ��̨�������
* @param[in]     void
* @param[out]    void
* @retval        void
* @par History   ��
*/
void right_detection()
{
	int i = 0;
	for(i = 0; i <= 15; i++) 						//����PWM��������Ч��ʱ�Ա�֤��ת����Ӧ�Ƕ�
	{
		Servo_J1(5);						//ģ�����PWM
	}
}

