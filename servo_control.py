import serial

def send_servo_command(ser, servo_num, angle):
    """
    发送舵机控制命令到串口
    :param ser: 串口对象
    :param servo_num: 舵机编号 (1-16)
    :param angle: 舵机角度 (0-180)
    """
    # 校验舵机编号范围
    if not 1 <= servo_num <= 16:
        raise ValueError("舵机编号需在 1-16 范围内")
    
    # 校验角度范围
    if not 0 <= angle <= 180:
        raise ValueError("角度需在 0-180 范围内")

    # 转换舵机编号为协议字符 (A-P)
    servo_char = chr(ord('A') + (servo_num - 1))
    
    # 格式化角度为3位字符串
    angle_str = f"{angle:03d}"
    
    # 构建完整命令
    command = f"${servo_char}{angle_str}#"
    
    # 发送命令到串口
    ser.write(command.encode('ascii'))
    print(f"[发送成功] 舵机{servo_num} → {angle}度 | 原始命令: {command}")

if __name__ == "__main__":
    import sys
    
    # 参数验证
    if len(sys.argv) != 3:
        print("用法: python servo_control.py <舵机编号(1-16)> <角度(0-180)>")
        print("示例: python servo_control.py 1 90")
        sys.exit(1)

    try:
        servo_num = int(sys.argv[1])
        angle = int(sys.argv[2])
    except ValueError:
        print("错误：参数必须为整数")
        sys.exit(1)

    try:
        # 配置串口（根据实际情况修改端口）
        with serial.Serial(
            port='COM10',  # Windows改为COMx，如COM3
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        ) as ser:
            send_servo_command(ser, servo_num, angle)
            
    except serial.SerialException as e:
        print(f"串口连接失败: {e}")
    except Exception as e:
        print(f"发生错误: {e}")