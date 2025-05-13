#include "AllHeader.h"


/******************************
*�������ܣ�����ѹ���쳣������\oled���� --������غ�ֻ�ܸ�λ�ָ�
*��ڲ�������
*����  ֵ����
******************************/
char buf[20]={'\0'};
void dectect_beep(void)
{ 
	u8 V_flag = 1;//��ѹ
	float vol;
	V_flag = Bsp_Bat_State(); 
	
	if(V_flag == 0)//��ѹ����� 
	{
		set_beep(5);//��ͣ����
		OLED_Draw_Line("voltage low...", 1, false, true);
		
		led_timer(); //�ƵĶ�ʱ�� --����
	}

	if(V_flag == 1)//���������
	{
		sprintf(buf,"Grop:%d  Ver:%.1f ",grop,version);
		OLED_Draw_Line(buf, 1 , false, true);
		memset(buf,0,sizeof(buf));
		vol = Bsp_Bat_Voltage_Z100()/100.0;
		sprintf(buf,"Battery vol: %.1f V ",vol);
		OLED_Draw_Line(buf, 3, false, true);
		
	}
	
	
	beep_start_stop();//���������� 
}



/******************************
*�������ܣ�������ĵ������쳣������\oled����
*��ڲ�������
*����  ֵ����
******************************/
//���Ӹú���
void void_jutce(void)
{
	u8 I_flag = 0;//����
	float junce;
	I_flag =J_Bsp_Bat_State(); 
	if(I_flag == 0)
	{
		junce = J_Bsp_Bat_Voltage_Z100()/100.0;
		sprintf(buf,"Current: %.2f A ",junce);
		
		OLED_Draw_Line(buf, 2, false, true);
	}
	else
	{
		junce = J_Bsp_Bat_Voltage_Z100()/100.0;
		sprintf(buf,"Current hard!: %.2f A ",junce);
		set_beep(5);//��ͣ����
		OLED_Draw_Line(buf, 2, false, true);
	}
	
	
	beep_start_stop();//���������� 
}

