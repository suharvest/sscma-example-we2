import serial,time,re,json,collections
s=serial.Serial("/dev/cu.wchusbserial58370593761",921600,timeout=0.5)
time.sleep(0.3); s.reset_input_buffer()
s.write(b"AT+MODEL=3\r\n"); s.flush(); time.sleep(4); s.reset_input_buffer()
s.write(b"AT+INVOKE=-1,0,1\r\n"); s.flush()
t0=time.time(); buf=b""
while time.time()-t0<90: buf+=s.read(65536)
s.write(b"AT+BREAK\r\n"); s.flush(); time.sleep(0.5); s.close()
txt=buf.decode("utf-8","replace")
open(SPD:=__file__.rsplit("/",1)[0]+"/live_verify.log","w").write(txt)

frames=[]
for _l in txt.replace('\r','\n').splitlines():
    _l=_l.strip()
    if _l.startswith('{"type": 1, "name": "INVOKE"'):
        try: frames.append(_l)
        except Exception: pass
print(f"帧数 {len(frames)}  时长 {time.time()-t0:.0f}s")
nbox=0; withp=0; tids=collections.Counter(); series=[]
maxcur=0
for f in frames:
    try: d=json.loads(f)["data"]
    except Exception: continue
    b=d.get("boxes",[]); c=d.get("counts",{})
    if b: withp+=1
    for x in b:
        nbox+=1
        if len(x)>=7: tids[x[6]]+=1
    ln=c.get("lines",[{}])[0] if c.get("lines") else {}
    ro=c.get("rois",[{}])[0] if c.get("rois") else {}
    maxcur=max(maxcur, ro.get("cur",0))
    series.append((ln.get("in",0), ln.get("out",0), ro.get("cur",0), ro.get("entered",0)))
print(f"有框帧 {withp}/{len(frames)} = {100*withp/max(1,len(frames)):.1f}%   总框数 {nbox}")
print(f"track_id 分布(次数): {dict(tids.most_common(12))}")
print(f"不同 track_id 个数: {len(tids)}")
if series:
    print(f"最终 counts: in={series[-1][0]} out={series[-1][1]} entered={series[-1][3]}  区域内峰值={maxcur}")
    prev=None; print("\n变化点:")
    for i,v in enumerate(series):
        if prev and v!=prev: print(f"  帧{i:4d}  in={v[0]} out={v[1]} cur={v[2]} entered={v[3]}")
        prev=v
