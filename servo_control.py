import serial
import sys

def send_servo_command(ser, servo_num, angle):
    # 参数校验与命令生成（同原有脚本）
    servo_char = chr(ord('A') + (servo_num - 1))
    command = f"${servo_char}{angle:03d}#"
    ser.write(command.encode('ascii'))
    print(f"已发送: {command}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python servo_control.py <舵机编号> <角度>")
        sys.exit(1)

    servo_num = int(sys.argv[1])
    angle = int(sys.argv[2])

    try:
        with serial.Serial(
            port='COM10',
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=2
        ) as ser:
            # 测试连接
            ser.write(b"TEST")
            response = ser.read(4)
            if response == b"TEST":
                print("蓝牙连接正常！")
                send_servo_command(ser, servo_num, angle)
            else:
                print("蓝牙无响应，请检查模块状态！")
                
    except serial.SerialException as e:
        print(f"串口错误: {e}")
        print("可能原因：端口不存在、被占用或权限不足")
    except Exception as e:
        print(f"未知错误: {e}")