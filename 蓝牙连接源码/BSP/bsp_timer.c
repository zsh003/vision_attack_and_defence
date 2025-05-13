/**
* @par Copyright (C): 2010-2019, Shenzhen Yahboom Tech
* @file         bsp_timer.h	
* @author       
* @version      V1.0
* @date         
* @brief        ��ʱ��
* @details      
* @par History  ������˵��
*                 
* version:		
*/

#include "AllHeader.h"



/**
* Function       TIM1_Int_Init
* @author        
* @date             
* @brief         ��ʱ��1��ʼ���ӿ�
* @param[in]     arr���Զ���װֵ��psc��ʱ��Ԥ��Ƶ��
* @param[out]    void
* @retval        void
* @par History   ����ʱ��ѡ��ΪAPB1��2������APB1Ϊ36M
*/
void TIM1_Int_Init(u16 arr,u16 psc)
{
    TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
	NVIC_InitTypeDef NVIC_InitStructure;

	RCC_APB2PeriphClockCmd(RCC_APB2Periph_TIM1, ENABLE); //ʱ��ʹ��
	
	//��ʱ��TIM1��ʼ��
	TIM_TimeBaseStructure.TIM_Period = arr; //��������һ�������¼�װ�����Զ���װ�ؼĴ������ڵ�ֵ	
	TIM_TimeBaseStructure.TIM_Prescaler = (psc-1); //����������ΪTIMxʱ��Ƶ�ʳ�����Ԥ��Ƶֵ
	TIM_TimeBaseStructure.TIM_ClockDivision = TIM_CKD_DIV1; //����ʱ�ӷָ�:TDTS = Tck_tim   //36Mhz
	TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;  //TIM���ϼ���ģʽ
	TIM_TimeBaseStructure.TIM_RepetitionCounter = 0;    //�ظ������ر�
	TIM_TimeBaseInit(TIM1, &TIM_TimeBaseStructure); //����ָ���Ĳ�����ʼ��TIMx��ʱ�������λ
 
	TIM_ITConfig(TIM1, TIM_IT_Update, ENABLE ); //ʹ��ָ����TIM1�ж�,��������ж�

	//�ж����ȼ�NVIC����
	NVIC_InitStructure.NVIC_IRQChannel = TIM1_UP_IRQn;  //TIM1�ж�
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;  //��ռ���ȼ�0��
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3;  //�����ȼ�3��
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE; //IRQͨ����ʹ��
	NVIC_Init(&NVIC_InitStructure);  //��ʼ��NVIC�Ĵ���


	TIM_Cmd(TIM1, ENABLE);  //ʹ��TIMx					 
}
/**
* Function       TIM1_Int_Init
* @author        
* @date             
* @brief         ��ʱ��1�жϷ������: ��Ҫ����6·�������
* @param[in]     arr���Զ���װֵ��psc��ʱ��Ԥ��Ƶ��
* @param[out]    void
* @retval        void
* @par History   ����ʱ��ѡ��ΪAPB1��2������APB1Ϊ36M
*/
int num = 0;
int times = 100;

void TIM1_UP_IRQHandler(void)   //TIM1�ж�
{
	if (TIM_GetITStatus(TIM1, TIM_IT_Update) != RESET)  //���TIM1�����жϷ������
	{
		TIM_ClearITPendingBit(TIM1, TIM_IT_Update);  //���TIM1�����жϱ�־ 
		num++;
	

		#ifdef USE_SERVO_J1
		if(num <= (Angle_J[0][0] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J1_PORT, Servo_J1_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J1_PORT, Servo_J1_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif	   	

		#ifdef USE_SERVO_J2
		if(num <= (Angle_J[0][1] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J2_PORT, Servo_J2_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J2_PORT, Servo_J2_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif

		#ifdef USE_SERVO_J3

		if(num <= (Angle_J[0][2] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J3_PORT, Servo_J3_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J3_PORT, Servo_J3_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif

		#ifdef USE_SERVO_J4
		if(num <= (Angle_J[0][3] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J4_PORT, Servo_J4_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J4_PORT, Servo_J4_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif

		#ifdef USE_SERVO_J5
		if(num <= (Angle_J[0][4] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J5_PORT, Servo_J5_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J5_PORT, Servo_J5_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif

		#ifdef USE_SERVO_J6
		if(num <= (Angle_J[0][5] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J6_PORT, Servo_J6_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J6_PORT, Servo_J6_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif
		
		#ifdef USE_SERVO_J7
		if(num <= (Angle_J[0][6] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J7_PORT, Servo_J7_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J7_PORT, Servo_J7_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif
		
		#ifdef USE_SERVO_J8
		if(num <= (Angle_J[0][7] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J8_PORT, Servo_J8_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J8_PORT, Servo_J8_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif
		
		#ifdef USE_SERVO_J9
		if(num <= (Angle_J[1][0] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J9_PORT, Servo_J9_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J9_PORT, Servo_J9_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif
		
		#ifdef USE_SERVO_J10
		if(num <= (Angle_J[1][1] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J10_PORT, Servo_J10_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J10_PORT, Servo_J10_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif
		
		#ifdef USE_SERVO_J11
		if(num <= (Angle_J[1][2] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J11_PORT, Servo_J11_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J11_PORT, Servo_J11_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif
		#ifdef USE_SERVO_J12
		if(num <= (Angle_J[1][3] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J12_PORT, Servo_J12_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J12_PORT, Servo_J12_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif
		
		#ifdef USE_SERVO_J13
		if(num <= (Angle_J[1][4] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J13_PORT, Servo_J13_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J13_PORT, Servo_J13_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif
		#ifdef USE_SERVO_J14
		if(num <= (Angle_J[1][5] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J14_PORT, Servo_J14_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J14_PORT, Servo_J14_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif
		
		#ifdef USE_SERVO_J15
		if(num <= (Angle_J[1][6] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J15_PORT, Servo_J15_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J15_PORT, Servo_J15_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif	
		
		#ifdef USE_SERVO_J16
		if(num <= (Angle_J[1][7] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J16_PORT, Servo_J16_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J16_PORT, Servo_J16_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif	
		
		#ifdef USE_SERVO_J17
		if(num <= (Angle_J[2][0] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J17_PORT, Servo_J17_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J17_PORT, Servo_J17_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif	
				
		#ifdef USE_SERVO_J18
		if(num <= (Angle_J[2][1] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J18_PORT, Servo_J18_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J18_PORT, Servo_J18_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif	
				
						
		#ifdef USE_SERVO_J19
		if(num <= (Angle_J[2][2] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J19_PORT, Servo_J19_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J19_PORT, Servo_J19_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif	
						
		#ifdef USE_SERVO_J20
		if(num <= (Angle_J[2][3] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J20_PORT, Servo_J20_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J20_PORT, Servo_J20_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif	
						
		#ifdef USE_SERVO_J21
		if(num <= (Angle_J[2][4] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J21_PORT, Servo_J21_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J21_PORT, Servo_J21_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif

		#ifdef USE_SERVO_J22
		if(num <= (Angle_J[2][5] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J22_PORT, Servo_J22_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J22_PORT, Servo_J22_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif

		#ifdef USE_SERVO_J23
		if(num <= (Angle_J[2][6] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J23_PORT, Servo_J23_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J23_PORT, Servo_J23_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif		
		
		#ifdef USE_SERVO_J24
		if(num <= (Angle_J[2][7] * 11 + 500)/times)
		{
			GPIO_SetBits(Servo_J24_PORT, Servo_J24_PIN );		//������ӿڵ�ƽ�ø�
		}
		else
		{
			GPIO_ResetBits(Servo_J24_PORT, Servo_J24_PIN );		//������ӿڵ�ƽ�ø�
		}
		#endif	

		if(num == 200) //200*100=20ms  20msһ������
		{
			num = 0;
		}
		
	}
}
