#include "AllHeader.h"

 
//------------����˵��---------------
//PS2������--------STM32F1����1����
//��->�� DI��ʼ
//    VCC     -----------    V3.3 
//    GND     -----------    GND  
//    MISO=DI��DAT      --------- PA6
//    MOSI=DO��CMD      ----------PA7
//    CS      -----------    PA4
//    CLK     -----------    PA5 
 
int PS2_LX,PS2_LY,PS2_RX,PS2_RY,PS2_KEY;


void user_douji(void)
{
	//ҡ����ʱ����
		PS2_LX=PS2_AnologData(PSS_LX);      
		PS2_LY=PS2_AnologData(PSS_LY);
		PS2_RX=PS2_AnologData(PSS_RX);
		PS2_RY=PS2_AnologData(PSS_RY);
	
		PS2_KEY=PS2_DataKey();
	
		switch(PS2_KEY)//��������
		{
			/**************************/
			case PSB_SELECT: //��s1������
					if(Angle_J[grop][0] >= 180)
							Angle_J[grop][0] = 180;
					else 
						Angle_J[grop][0] +=10;//��10�ȵ����ϼ�
					break;
			case PSB_START: //��s1�ϼ���
				if(Angle_J[grop][0] <= 0)
							Angle_J[grop][0] = 0;
					else 
						Angle_J[grop][0] -=10;//��10�ȵ����¼�
				  break;
					
			/**************************/		
			case PSB_L3: //��s2������
					if(Angle_J[grop][1] >= 180)
							Angle_J[grop][1] = 180;
					else 
						Angle_J[grop][1] +=10;//��10�ȵ����ϼ�
					break;
			case PSB_R3: //��s2�ϼ���
					if(Angle_J[grop][1] <= 0)
							Angle_J[grop][1] = 0;
					else 
						Angle_J[grop][1] -=10;//��10�ȵ����¼�
				  break;
					
			/**************************/
			case PSB_PAD_UP: //��s3������
					if(Angle_J[grop][2] >= 180)
							Angle_J[grop][2] = 180;
					else 
						Angle_J[grop][2] +=10;//��10�ȵ����ϼ�
					break;
			case PSB_PAD_DOWN: //��s3�ϼ���
					if(Angle_J[grop][2] <= 0)
							Angle_J[grop][2] = 0;
					else 
						Angle_J[grop][2] -=10;//��10�ȵ����¼�
				  break;
					
			/**************************/
			case PSB_PAD_RIGHT: //��s4������
					if(Angle_J[grop][3] >= 180)
							Angle_J[grop][3] = 180;
					else 
						Angle_J[grop][3] +=10;//��10�ȵ����ϼ�
					break;
			case PSB_PAD_LEFT: //��s4�ϼ���
					if(Angle_J[grop][3] <= 0)
							Angle_J[grop][3] = 0;
					else 
						Angle_J[grop][3] -=10;//��10�ȵ����¼�
				  break;
					
			/**************************/
			case PSB_L2: //��s5������
					if(Angle_J[grop][4] >= 180)
							Angle_J[grop][4] = 180;
					else 
						Angle_J[grop][4] +=10;//��10�ȵ����ϼ�
					break;				
			case PSB_R2: //��s5�ϼ���
					if(Angle_J[grop][4] <= 0)
							Angle_J[grop][4] = 0;
					else 
						Angle_J[grop][4] -=10;//��10�ȵ����¼�
					break;				
			
			/**************************/
			case PSB_L1: //��s6������
					if(Angle_J[grop][5] >= 180)
							Angle_J[grop][5] = 180;
					else 
						Angle_J[grop][5] +=10;//��10�ȵ����ϼ�
					break;				
			case PSB_R1: //��s6�ϼ���
					if(Angle_J[grop][5] <= 0)
							Angle_J[grop][5] = 0;
					else 
						Angle_J[grop][5] -=10;//��10�ȵ����¼�
					break;		
			
			/**************************/
			case PSB_GREEN: //��s7������
					if(Angle_J[grop][6] >= 180)
							Angle_J[grop][6] = 180;
					else 
						Angle_J[grop][6] +=10;//��10�ȵ����ϼ�
					break;		
			case PSB_BLUE: //��s7�ϼ���
					if(Angle_J[grop][6] <= 0)
							Angle_J[grop][6] = 0;
					else 
						Angle_J[grop][6] -=10;//��10�ȵ����¼�
					break;		
					
			/**************************/
			case PSB_RED: //��s8������
					if(Angle_J[grop][7] >= 180)
							Angle_J[grop][7] = 180;
					else 
						Angle_J[grop][7] +=10;//��10�ȵ����ϼ�
					break;		
			case PSB_PINK: //��s8�ϼ���
					if(Angle_J[grop][7] <= 0)
							Angle_J[grop][7] = 0;
					else 
						Angle_J[grop][7] -=10;//��10�ȵ����¼�
					break;		
		}

}


//�����ֱ��ĳ���
#if 0
void test_ps2(void)
{ 
	while(1)
		{		
			PS2_LX=PS2_AnologData(PSS_LX);      
			PS2_LY=PS2_AnologData(PSS_LY);
			PS2_RX=PS2_AnologData(PSS_RX);
			PS2_RY=PS2_AnologData(PSS_RY);
			PS2_KEY=PS2_DataKey();	
 		  printf("%d     PS2��X��:",PS2_LX);   //��ӡ���X���ֵ
			printf("%d     PS2��Y��:",PS2_LY);   //��ӡ�ұ�Y���ֵ
		  printf("%d     PS2��X��:",PS2_RX);   //��ӡ���X���ֵ
			printf("%d     PS2��Y��:",PS2_RY);   //��ӡ�ұ�Y���ֵ
			printf("%d \r\nPS2����:",PS2_KEY);   //��ӡ������ֵ
			delay_ms(100);										   //��ʱ100m����
			switch(PS2_KEY)
			{
				case 5:
				case 6:
				case 7:
				case 8:
					BEEP_ON;
					break;
				
				case 9:
				case 10:
				case 11:
				case 12:
				case 13:
				case 14:
				case 15:
					BEEP_OFF;
					break;
				
			}
			delay_ms(150);
		} 
}
#endif



