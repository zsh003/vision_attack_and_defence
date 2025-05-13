#ifndef __AllHeader_H
#define __AllHeader_H


#define DEBUG_USARTx USART1  //��ӡ��Ϣ����

#define bool _Bool
#define true 1
#define false 0

#include <stdio.h>
#include "string.h"
#include "stm32f10x.h"
#include "bsp.h"
#include "stm32f10x_gpio.h"
#include "stm32f10x_pwr.h"
#include "stm32f10x_bkp.h"
#include "bsp_gpio.h"
#include "bsp_servo.h"
#include "bsp_ps2.h"
#include "delay.h"
#include "bsp_timer.h"
#include "bsp_beep.h"
#include "bsp_adc.h"
#include "bsp_juice_adc.h"
#include "bsp_usart.h"
#include "bsp_common.h"
#include "oled_iic.h"
#include "oled_fonts.h"
#include "bsp_oled.h"
#include "bsp_key.h"


#define GROUP_NUM 3 //24·�����3�飬һ��8��
#define DUOJI_NUM 8  //����ÿ��Ķ����
extern u8 grop ;//���������������ҪӦ�����ֱ�
extern float version;//�汾��


/*������Ҫ��ʼ���Ķ���궨�忪��*/
//1-8����һ�飩
#define USE_SERVO_J1
#define USE_SERVO_J2
#define USE_SERVO_J3
#define USE_SERVO_J4
#define USE_SERVO_J5
#define USE_SERVO_J6
#define USE_SERVO_J7
#define USE_SERVO_J8

//9-16���ڶ��飩
#define USE_SERVO_J9
#define USE_SERVO_J10
#define USE_SERVO_J11
#define USE_SERVO_J12
#define USE_SERVO_J13
#define USE_SERVO_J14
#define USE_SERVO_J15
#define USE_SERVO_J16

//16-24�������飩
#define USE_SERVO_J17
#define USE_SERVO_J18
#define USE_SERVO_J19
#define USE_SERVO_J20
#define USE_SERVO_J21
#define USE_SERVO_J22
#define USE_SERVO_J23
#define USE_SERVO_J24

/*ps2 �ֱ��Ķ��忪�� 1���� 0���ر�*/
#define PS_TWO 0

/*�������Ķ��忪�� 1���� 0���ر�*/
#define BEEP_SW 1

/*��ѹ������ⶨ�忪�� 1���� 0���ر�*/
#define Battery_SW 1

/*oled�Ķ��忪�� 1���� 0���ر�*/
#define OLED_SW 1




//��ʱ�����pwm��ʵ�ֽǶȿ���
extern int Angle_J[GROUP_NUM][DUOJI_NUM];

void void_jutce(void);

void dectect_beep(void);
void test_ps2(void);
void user_douji(void);
void deal_bluetooth(u8 Rx_temp);
u8 deal_data(void);
#endif


