import serial, time, re
s=serial.Serial("/dev/cu.wchusbserial58370593761",921600,timeout=0.5)
s.write(b"AT+RST\r\n"); s.flush()
t=time.time(); b=b""
while time.time()-t<7: b+=s.read(65536)
boot=b.decode("utf-8","replace")
print("=== BOOT 关键行 ===")
for l in [x for x in boot.splitlines() if x.strip()][:14]: print("  ",l[:160])
s.reset_input_buffer()
def cmd(c,w=2.5,n=4,cap=1500):
    s.write(c.encode()+b"\r\n"); s.flush()
    t=time.time(); d=b""
    while time.time()-t<w: d+=s.read(65536)
    txt=d.decode("utf-8","replace")
    print(f"\n### {c}")
    for l in [x for x in txt.splitlines() if x.strip()][:n]: print("  ",l[:cap])
    return txt
cmd("AT+CNTCFG?")
cmd("AT+CNTLINE?")
cmd("AT+CNTROI?")
# 配一条水平中线 + 一个中央方形区域（归一化 0..1000）
cmd("AT+CNTLINE=0,0,500,1000,500")
cmd("AT+CNTROI=0,250,250,750,250,750,750,250,750")
cmd("AT+CNTLINE?")
cmd("AT+CNTROI?")
s.close()
