import serial,time,re,base64,json
s=serial.Serial("/dev/cu.wchusbserial58370593761",921600,timeout=0.5)
time.sleep(0.3); s.reset_input_buffer()
s.write(b"AT+SAMPLE=1\r\n"); s.flush()
t=time.time(); d=b""
while time.time()-t<12: d+=s.read(65536)
txt=d.decode("utf-8","replace")
m=re.findall(r'"image": "([A-Za-z0-9+/=]+)"', txt)
print("找到 image 字段:", len(m), "长度:", [len(x) for x in m][:3])
if m:
    raw=base64.b64decode(m[-1])
    p=__file__.rsplit("/",1)[0]+"/camera_view.jpg"
    open(p,"wb").write(raw); print("saved", p, len(raw), "B")
else:
    print(txt[:600])
s.close()
