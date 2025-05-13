#include "AllHeader.h"

#define RX_LEN 10

char rxbuff[RX_LEN];//һ�����ݵĴ洢��
u8 recv_i = 0;
u8 rx_flag = 0;//��Ч���ݰ���λ��
u8 newdata_flag = 0;//���һ�����ݽ��յı�־

//������������Ϣ��������  //$A090#  :��һ·�Ķ��90��
void deal_bluetooth(u8 Rx_temp)  //��β��#��ȥ��
{
		if(Rx_temp == '$')
	{
		rx_flag = 1;
	}
	if(Rx_temp == '#')
	{
		rxbuff[recv_i] = '\0';
		recv_i = 0;
		rx_flag = 0;
		deal_data();
		memset(rxbuff,0,RX_LEN);
	}
	if(rx_flag == 1 )
	{
		rxbuff[recv_i] = Rx_temp;
		recv_i ++;
	}

}

//�����������
u8 deal_data(void) //�����������ַ�����
{
	u8 duo_num = 0;
	u8 Hor,column;
	u16 angle = 0;
	duo_num = rxbuff[1] - 64; 
	Hor = (duo_num-1)/8;//�õ���� 0.1.2     -1:�����Ǵ�0��ʼ
	column=(duo_num-1)%8; //һ����Ķ�Ӧ�Ķ��λ��
	
	angle = (rxbuff[2]-48)*100 + (rxbuff[3]-48)*10 + rxbuff[4]-48;
	
	if(angle <= 0)
	{
		angle = 0;
	}
	if(angle >= 180)
	{
		angle = 180;
	}
	
	Angle_J[Hor][column] = angle;	
	
	return 0;
}

