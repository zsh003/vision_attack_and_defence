#include "AllHeader.h"

u32 beep_time = 0;
u32 led_time = 0;
u8 led_state = 1;


void init_led(void)
{
	//��ʼ��������������
  GPIO_InitTypeDef GPIO_InitStructure;
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(LED_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = LED_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��BEEP_PORT*/
  	GPIO_Init(LED_PORT, &GPIO_InitStructure);
	
	GPIO_SetBits(LED_PORT,LED_PIN);
	
	

}

void led_timer(void) //��ʱ��ʱ�� (999+1)*(720+1)/72M = 10ms 
{
	TIM_TimeBaseInitTypeDef TIM_TimeBaseInitStructer;
	NVIC_InitTypeDef NVIC_InitStructer;

	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM3, ENABLE);
	
	/*��ʱ��TIM3��ʼ��*/
	TIM_DeInit(TIM3);
	TIM_TimeBaseInitStructer.TIM_Period = 1000-1;//��ʱ����
	TIM_TimeBaseInitStructer.TIM_Prescaler = 720-1; //��Ƶϵ��
	TIM_TimeBaseInitStructer.TIM_ClockDivision = TIM_CKD_DIV1;//����Ƶ
	TIM_TimeBaseInitStructer.TIM_CounterMode = TIM_CounterMode_Up;
	TIM_TimeBaseInit(TIM3, &TIM_TimeBaseInitStructer);
	
	TIM_ITConfig(TIM3, TIM_IT_Update, ENABLE);//���������ж�

	/*��ʱ���жϳ�ʼ��*/
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
	
	NVIC_InitStructer.NVIC_IRQChannelPreemptionPriority = 0;
	NVIC_InitStructer.NVIC_IRQChannelSubPriority = 1;
	NVIC_InitStructer.NVIC_IRQChannel = TIM3_IRQn;
	NVIC_InitStructer.NVIC_IRQChannelCmd = ENABLE;
	
	NVIC_Init(&NVIC_InitStructer);
	TIM_Cmd(TIM3, ENABLE);//��ʱ��ʹ��
	
}

void TIM3_IRQHandler(void) //�жϣ��������źźܳ��ǣ�����ֵ������ظ����������ж��������������
{
	if(TIM_GetITStatus(TIM3,TIM_IT_Update) != RESET)
	{
		TIM_ClearITPendingBit(TIM3, TIM_IT_Update);//����жϱ�־
		
		led_time ++;
//3��������
		if(led_time % 60 ==0)
		{
			led_state = !led_state;
			LED(led_state);
		}
		
		if(led_time >=300 )
		{
			led_time = 0;
		}
		
	}
}


void init_beep(void)
{
	//��ʼ��������������
  GPIO_InitTypeDef GPIO_InitStructure;
	/*��������ʱ��*/
	RCC_APB2PeriphClockCmd(BEEP_RCC, ENABLE); 
	/*ѡ��Ҫ���Ƶ�����*/															   
  	GPIO_InitStructure.GPIO_Pin = BEEP_PIN;	
	/*��������ģʽΪͨ���������*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   
	/*������������Ϊ50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	/*���ÿ⺯������ʼ��BEEP_PORT*/
  	GPIO_Init(BEEP_PORT, &GPIO_InitStructure);
		
	beep_timer();//��ʼ����ʱ��

}

void beep_timer(void) //��ʱ��ʱ�� (99+1)*(720+1)/72M = 1ms 
{
	TIM_TimeBaseInitTypeDef TIM_TimeBaseInitStructer;
	NVIC_InitTypeDef NVIC_InitStructer;

	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM2, ENABLE);
	
	/*��ʱ��TIM2��ʼ��*/
	TIM_DeInit(TIM2);
	TIM_TimeBaseInitStructer.TIM_Period = 100-1;//��ʱ����
	TIM_TimeBaseInitStructer.TIM_Prescaler = 720-1; //��Ƶϵ��
	TIM_TimeBaseInitStructer.TIM_ClockDivision = TIM_CKD_DIV1;//����Ƶ
	TIM_TimeBaseInitStructer.TIM_CounterMode = TIM_CounterMode_Up;
	TIM_TimeBaseInit(TIM2, &TIM_TimeBaseInitStructer);
	
	TIM_ITConfig(TIM2, TIM_IT_Update, ENABLE);//���������ж�

	/*��ʱ���жϳ�ʼ��*/
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
	
	NVIC_InitStructer.NVIC_IRQChannelPreemptionPriority = 0;
	NVIC_InitStructer.NVIC_IRQChannelSubPriority = 0;
	NVIC_InitStructer.NVIC_IRQChannel = TIM2_IRQn;
	NVIC_InitStructer.NVIC_IRQChannelCmd = ENABLE;
	
	NVIC_Init(&NVIC_InitStructer);
	TIM_Cmd(TIM2, ENABLE);//��ʱ��ʹ��
}

void TIM2_IRQHandler(void) //�жϣ��������źźܳ��ǣ�����ֵ������ظ����������ж��������������
{
	if(TIM_GetITStatus(TIM2,TIM_IT_Update) != RESET)
	{
		TIM_ClearITPendingBit(TIM2, TIM_IT_Update);//����жϱ�־
		if(beep_time != 0)
				beep_time -- ;
	}
}


/**
* Function       
* @brief         ���÷��������ʱ��
* @param[in]     
* @param[out]    times:���ʱ��
* @retval        void
* @par History   ��
*/
void set_beep(u8 times) //sΪ��λ
{
	beep_time = times * 1000;
}

/* ���º����Ž���ѭ�� */
//------------------------------
//����������������Ƶ���죬TIM2�ж�2�����һ������ 1s��5��
void beep_start_stop(void)
{
	if(beep_time != 0)
	{
		BEEP_ON;
		delay_ms(100);
		BEEP_OFF;
		delay_ms(100);
	}
	
	else if(beep_time == 0)
		BEEP_OFF;
}
//----------------------------


