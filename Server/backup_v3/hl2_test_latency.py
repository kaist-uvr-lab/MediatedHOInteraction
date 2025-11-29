import socket

host = "0.0.0.0"  # 모든 인터페이스에서 수신
port = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((host, port))

print(f"Listening on {host}:{port}")

while True:
    data, addr = sock.recvfrom(1024)  # 유니티에서 보낸 데이터 수신
    # 그대로 돌려보냄 (Echo)
    sock.sendto(data, addr)