#include "AllHeader.h"

u8 grop = 0;
float version = 1.1;


int main(void)
{	
	u8 Dnum;
	u8 Doup;

	//����Ƕȳ�ʼ��
	for(Doup = 0;Doup <GROUP_NUM;Doup++)
	{  
		for(Dnum = 0;Dnum <DUOJI_NUM;Dnum++)
		{
			Angle_J[Doup][Dnum] = 90;
		}
		
	}
	
	bsp_init();//�ٽ���Ӳ����ʼ�� 
	
	while (1)
	{
		dectect_beep();//����ѹ
		
		void_jutce(); //������

#if	PS_TWO
		user_douji();	//�ֱ�����
#endif		
		if(Key1_State(1)==1)//����ֻ��Чһ��
		{
			grop++;
			if(grop>2)
			{
				grop = 0; 
			}
				
		}
		
	}
 								    
}


