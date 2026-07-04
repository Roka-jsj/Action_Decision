import torch, socket
print("cuda:", torch.cuda.is_available(), "n=", torch.cuda.device_count())
try:
    socket.create_connection(("pypi.org", 443), timeout=5); print("internet: OK")
except Exception as e:
    print("internet: BLOCKED", e)
