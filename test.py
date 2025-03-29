import sensor, image, time, lcd, network, usocket, ustruct
from maix import KPU


s=usocket.socket(af=AF_INET, type=SOCK_STREAM, proto=IPPROTO_TCP)
address = ('192.168.124.29', 5000)  # PC 端 IP
s.connect(address)
