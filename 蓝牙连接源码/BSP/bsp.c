/**
* @par Copyright (C): 2010-2019, Shenzhen Yahboom Tech
* @file         bsp.c      
* @version      V1.0     
* @brief        ���������
* @details      
* @par History  ������˵��
*                 
* version:	
*/


#include "AllHeader.h"

/**
* Function       bsp_init  
* @brief         Ӳ���豸��ʼ��
* @param[in]     void
* @param[out]    void
* @retval        void
* @par History   ��
*/
void bsp_init(void)
{
	SystemInit();
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2); 
	
	delay_init();
	delay_ms(200);//�ȴ�ϵͳ�ȶ�
	

	
	Servo_GPIO_Init();
	TIM1_Int_Init(99, 72);			/*������10Ϊ100us   Tout = (99+1)*(71+1)/72M = 100us */
////	
	USART1_init(115200); //���Դ���
	USART3_init(9600);//����λ��ͨ��
	
	UART5_init(9600);//---����ͨ��Ĭ�ϲ�����Ϊ9600

	init_led(); //�Ƴ�ʼ��
	init_beep();  //��������ʼ��
	Key_GPIO_Init();//������ʼ��
  
	
	
#if	Battery_SW
	Adc_vol_init();  //����ѹ
	J_Adc_init();//������
#endif	

	
#if	PS_TWO
	PS2_Init();		//======ps2�����˿ڳ�ʼ��
	PS2_SetInit();	//======ps2���ó�ʼ��,���á����̵�ģʽ������ѡ���Ƿ�����޸�
	delay_ms(1000);
#endif
	
#if OLED_SW
	IIC_Init();
	i2c_scanf_addr();
	SSD1306_Init();
		OLED_Draw_Line("OLED init sucess", 1 , false, true);
//	OLED_Draw_Line("Servo angle:90 ", 1 , false, true);
//	OLED_Draw_Line("current:0.03  A", 2 , false, true);
//	OLED_Draw_Line("Battery vol: 7.2 V", 3 , false, true);
	

#endif

	//�ŵ�������Ч����Ȼ�����޷�����ʹ��
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO, ENABLE);
	GPIO_PinRemapConfig(GPIO_Remap_SWJ_JTAGDisable, ENABLE);//����jlink ֻ��SWD���Կڣ�PA15��PB3��4����ͨIO
	

}
