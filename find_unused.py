import socket

def find_unused_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

unused_port = find_unused_port()
print(f"Available port: {unused_port}")
