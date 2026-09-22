"""Explicit live-only adapters for the historical three-axis experiment rig."""
from __future__ import annotations
from collections import deque
import re
import struct
import threading
import time
import numpy as np
from demo_core import AXES, SENSORS, DemoError, summarise

def parse_state(text, cfg, require_torque=True):
    states = {}
    for line in text.splitlines():
        fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
        if 'axis' in fields and 'raw_pos' in fields and 'mapping_ready' in fields:
            if fields['axis'] in states:
                raise DemoError('Duplicate STATE axis')
            states[fields['axis']] = fields
    if set(states) != set(AXES):
        raise DemoError(f'Expected historical 3-axis rig only, received {list(states)}; 5-axis firmware is incompatible')
    for j, axis in enumerate(AXES):
        f = states[axis]
        try:
            if int(f['id']) != cfg['hardware']['servo_ids'][j]:
                raise DemoError(f'{axis}: servo ID mismatch')
            if f['ping'] != 'true' or f['mapping_ready'] != 'true' or f['fault'] != 'false':
                raise DemoError(f'{axis}: ping/mapping/fault guard failed')
            if require_torque and f['torque'] != 'true':
                raise DemoError(f'{axis}: torque lost')
            if int(f['read_failures']) or int(f['overload_samples']):
                raise DemoError(f'{axis}: read/overload error')
            if f['direction'] != 'normal':
                raise DemoError(f'{axis}: changed direction')
            actual = [int(f['raw_active_min']), int(f['raw_active_max'])]
            if actual != cfg['hardware']['raw_limits'][j]:
                raise DemoError(f'{axis}: calibration range changed: {actual}')
            pos = int(f['raw_pos'])
            if not actual[0] <= pos <= actual[1]:
                raise DemoError(f'{axis}: position outside active range')
            for name in ('raw_speed','last_update_ms'):
                int(f[name])
        except (KeyError, ValueError) as exc:
            raise DemoError(f'{axis}: incomplete or malformed STATE') from exc
    return states

class Motor:
    def __init__(self, cfg, log):
        import serial
        self.cfg, self.log = cfg, log
        self.last_updates = {}
        self.before_move = lambda: None
        self.engaged = False
        self.ser = serial.Serial(cfg['hardware']['esp32_port'], 115200, timeout=.05, write_timeout=1)
        time.sleep(2.0)  # Board may reset. Operator supports the arm before opening.

    def request(self, command):
        self.ser.reset_input_buffer()
        self.ser.write((command+'\r\n').encode('ascii'))
        self.ser.flush()
        self.log('command', command=command)
        received = bytearray()
        deadline = time.monotonic()+2
        last_byte = time.monotonic()
        while time.monotonic()<deadline:
            chunk = self.ser.read(max(1, self.ser.in_waiting))
            if chunk:
                received.extend(chunk)
                last_byte=time.monotonic()
            elif received and time.monotonic()-last_byte>.2:
                break
        text=received.decode('ascii', errors='replace')
        self.log('response', command=command, text=text)
        if not text.strip() or re.search(r'blocked|failed|invalid|usage:|FAULT_LATCHED',text,re.I):
            raise DemoError(f'No successful response to {command}: {text}')
        return text

    def state(self, require_torque=True):
        states=parse_state(self.request('STATE'), self.cfg, require_torque)
        for axis, f in states.items():
            update=int(f['last_update_ms'])
            if update <= self.last_updates.get(axis,-1):
                raise DemoError(f'{axis}: stale telemetry or board reset')
            self.last_updates[axis]=update
        return states

    @staticmethod
    def positions(states):
        return np.array([int(states[a]['raw_pos']) for a in AXES])

    def command_axis(self, axis, value, health=True):
        if health:
            self.before_move()
        j=AXES.index(axis)
        lo,hi=self.cfg['hardware']['raw_limits'][j]
        if not lo <= int(value) <= hi:
            raise DemoError('Out-of-range MOVE_RAW refused before write')
        speed=self.cfg['hardware']['speed']
        acc=self.cfg['hardware']['acceleration']
        reply=self.request(f'MOVE_RAW {axis} {int(value)} {speed} {acc}')
        matched=re.search(r'\[MOVE_RAW\] axis=(\w+) id=(\d+) raw=(-?\d+)',reply)
        if not matched or matched[1]!=axis or int(matched[3])!=int(value) or int(matched[2])!=self.cfg['hardware']['servo_ids'][j]:
            raise DemoError('MOVE_RAW acknowledgement missing or clamped')

    def safe_engage(self):
        # Read rig identity BEFORE any torque or calibration command.
        # Mapping may not be loaded after reset: inspect axis/IDs without accepting motion.
        reply=self.request('STATE')
        ids=dict(re.findall(r'axis=(\w+) alias=\S+ id=(\d+)',reply))
        if ids != dict(zip(AXES, map(str,self.cfg['hardware']['servo_ids']))):
            raise DemoError('Unexpected rig, no torque/calibration commands sent')
        off=self.request('TORQUE OFF ALL')
        if any(f'[TORQUE] axis={a} state=OFF' not in off for a in AXES):
            raise DemoError('Explicit torque-off acknowledgement incomplete')
        self.request('CAL_AUTO LOAD')
        states=self.state(require_torque=False)
        start=self.positions(states)
        for axis,pos in zip(AXES,start):
            self.command_axis(axis,int(pos))
        after=self.positions(self.state(require_torque=False))
        if np.max(np.abs(after-start))>3:
            raise DemoError('Supported arm moved while preloading goals; re-support before retry')
        # All three goals now match current positions, as in original safe_engage.
        on=self.request('TORQUE ON ALL')
        self.engaged=True
        if any(f'[TORQUE] axis={a} state=ON' not in on for a in AXES):
            raise DemoError('Torque-on acknowledgement incomplete')
        if np.max(np.abs(self.positions(self.state())-start))>10:
            raise DemoError('Unexpected movement during safe_engage')

    def wait(self, seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            self.before_move()
            self.state()
            time.sleep(.05)

    def move(self, destination, force_goal=False):
        dest=np.asarray(destination,dtype=int)
        if dest.shape!=(3,):
            raise DemoError('3-axis target required')
        limits=np.asarray(self.cfg['hardware']['raw_limits'])
        if np.any(dest<limits[:,0]+40) or np.any(dest>limits[:,1]-40):
            raise DemoError('Destination outside joint margin')
        cfg=self.cfg['control']
        # No long queued trajectory: one bounded step, wait for arrival, then next.
        for j,axis in enumerate(AXES):
            attempts=0
            while True:
                self.before_move()
                state=self.state()
                pos=int(state[axis]['raw_pos'])
                if abs(dest[j]-pos)<=cfg['arrival_tolerance_ticks']:
                    break
                attempts+=1
                if attempts>100:
                    raise DemoError('Transit exceeded step budget')
                next_pos=int(pos+np.clip(dest[j]-pos,-cfg['max_transit_step_ticks'],cfg['max_transit_step_ticks']))
                self.command_axis(axis,next_pos)
                deadline=time.monotonic()+cfg['settle_timeout_s']
                stable=0
                while time.monotonic()<deadline:
                    self.before_move()
                    state=self.state()
                    arrived=abs(int(state[axis]['raw_pos'])-next_pos)<=cfg['arrival_tolerance_ticks']
                    stable=stable+1 if arrived and abs(int(state[axis]['raw_speed']))<=5 else 0
                    if stable>=3:
                        break
                    time.sleep(.05)
                if stable<3:
                    raise DemoError(f'{axis}: arrival timeout; remaining moves cancelled')
            if force_goal:
                # Arrival tolerance is feedback tolerance, not permission to
                # discard a 1-3 tick correction goal.
                self.state()
                self.command_axis(axis,int(dest[j]))
            self.wait(cfg['axis_gap_s'])
        arrived=self.positions(self.state())
        if np.max(np.abs(arrived-dest))>cfg['arrival_tolerance_ticks']:
            raise DemoError('Final joint arrival check failed')

    def hold_after_abort(self):
        # Best effort only. A disconnected bus cannot provide software stopping.
        # Never fault-reset, re-enable torque, park blindly, or drop the arm on error.
        if not self.engaged:
            return
        try:
            positions=self.positions(self.state())
            for axis,pos in zip(AXES,positions):
                self.command_axis(axis,int(pos),health=False)
            self.log('abort_hold', status='current_goals_written_torque_retained')
        except Exception as exc:
            self.log('abort_hold_failed', error=str(exc))
            print('SOFTWARE HOLD FAILED. Use the external stop and support the arm.')

    def release_supported(self):
        response=self.request('TORQUE OFF ALL')
        if any(f'[TORQUE] axis={a} state=OFF' not in response for a in AXES):
            raise DemoError('Torque-off not fully acknowledged')
        self.engaged=False

    def close(self):
        self.ser.close()

def extract_packets(buffer):
    packets=[]
    discarded=0
    while len(buffer)>=11:
        if buffer[0]!=0x55 or sum(buffer[:10]) & 255 != buffer[10]:
            del buffer[0]
            discarded+=1
            continue
        packet=bytes(buffer[:11]); del buffer[:11]
        values=np.asarray(struct.unpack('<hhhh',packet[2:10])[:3],float)
        kind=packet[1]
        if kind==0x51:
            packets.append(('acceleration',values/32768*16*9.80665,packet.hex()))
        elif kind==0x52:
            packets.append(('angular_velocity',values/32768*np.deg2rad(2000),packet.hex()))
        elif kind==0x53:
            packets.append(('angle',values/32768*np.pi,packet.hex()))
    return packets,discarded

class IMUs:
    def __init__(self,cfg,log):
        import serial
        self.cfg,self.log=cfg,log
        self.lock=threading.Lock()
        self.stop=threading.Event()
        self.failures=[]
        self.samples={s:{k:deque(maxlen=1000) for k in ('angle','acceleration','angular_velocity')} for s in SENSORS}
        self.ports={}
        self.threads=[]
        try:
            for sensor in SENSORS:
                port=serial.Serial(cfg['hardware']['imu_ports'][sensor],9600,timeout=.1)
                self.ports[sensor]=port
                port.reset_input_buffer()
                thread=threading.Thread(target=self.reader,args=(sensor,port),daemon=True)
                self.threads.append(thread)
                thread.start()
        except BaseException:
            self.close()
            raise

    def reader(self,sensor,port):
        buffer=bytearray()
        synchronized=False
        try:
            while not self.stop.is_set():
                buffer.extend(port.read(max(1,port.in_waiting)))
                packets,discarded=extract_packets(buffer)
                if discarded and synchronized:
                    raise DemoError(f'{sensor}: checksum/synchronization loss')
                if packets:
                    synchronized=True
                now=time.monotonic()
                for kind,value,raw_hex in packets:
                    with self.lock:
                        self.samples[sensor][kind].append((now,value))
                    self.log('imu_raw',sensor=sensor,packet_type=kind,values=value.tolist(),raw_hex=raw_hex)
        except Exception as exc:
            with self.lock:
                self.failures.append(str(exc))

    def ensure_alive(self):
        with self.lock:
            if self.failures:
                raise DemoError('; '.join(self.failures))
            now=time.monotonic()
            for s in SENSORS:
                for k,values in self.samples[s].items():
                    if not values or now-values[-1][0]>.5:
                        raise DemoError(f'{s}/{k}: missing or stale stream')

    def snapshot(self,start,end):
        self.ensure_alive()
        with self.lock:
            result={}
            for s in SENSORS:
                result[s]={}
                for k,values in self.samples[s].items():
                    selected=[(t,v) for t,v in values if start<=t<=end]
                    times=[start,*[t for t,_ in selected],end]
                    if max(np.diff(times))>.5:
                        raise DemoError(f'{s}/{k}: gap within summary window')
                    result[s][k]=[v.tolist() for _,v in selected]
            return result

    def settled(self):
        self.ensure_alive()
        now=time.monotonic()
        with self.lock:
            for s in SENSORS:
                v=[value for t,value in self.samples[s]['angular_velocity'] if now-t<=1.]
                if len(v)<6 or np.sqrt(np.mean(np.sum(np.asarray(v)**2,axis=1)))>self.cfg['control']['max_gyro_rms_rad_s']:
                    return False
        return True

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=1)
        for port in self.ports.values():
            port.close()

class ViveAudit:
    """Optional asynchronous label recorder; never supplies control features."""
    def __init__(self,serial,log):
        import openvr
        self.openvr,self.log=openvr,log
        self.system=openvr.init(openvr.VRApplication_Other)
        self.index=None
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if self.system.getTrackedDeviceClass(i)==openvr.TrackedDeviceClass_GenericTracker and self.system.getStringTrackedDeviceProperty(i,openvr.Prop_SerialNumber_String)==serial:
                self.index=i
        if self.index is None:
            openvr.shutdown()
            raise DemoError('Requested Vive audit serial not found')
        self.lock=threading.Lock(); self.samples=deque(maxlen=6000)
        self.stop=threading.Event(); self.error=None
        self.thread=threading.Thread(target=self.reader,daemon=True); self.thread.start()

    def reader(self):
        try:
            while not self.stop.is_set():
                poses=self.system.getDeviceToAbsoluteTrackingPose(self.openvr.TrackingUniverseStanding,0,self.openvr.k_unMaxTrackedDeviceCount)
                p=poses[self.index]; now=time.monotonic()
                valid=bool(p.bPoseIsValid and p.bDeviceIsConnected)
                xyz=[float(p.mDeviceToAbsoluteTracking[i][3])*1000 for i in range(3)] if valid else None
                with self.lock:
                    self.samples.append((now,xyz))
                self.log('vive_audit',valid=valid,world_mm=xyz)
                self.stop.wait(1/60)
        except Exception as exc:
            self.error=str(exc)

    def summary(self,start,end):
        with self.lock:
            samples=[(t,v) for t,v in self.samples if start<=t<=end]
        good=[v for _,v in samples if v is not None]
        gaps=np.diff([start,*[t for t,_ in samples],end])
        valid=len(good)>=30 and len(good)/max(1,len(samples))>=.99 and max(gaps)<.2 and self.error is None
        return {'valid':valid,'n':len(samples),'valid_n':len(good),'error':self.error,
                'measured_world_mm':np.mean(good,axis=0).tolist() if valid else None}

    def close(self):
        self.stop.set(); self.thread.join(timeout=2); self.openvr.shutdown()

def measure(motor,imus,cfg,audit=None):
    deadline=time.monotonic()+cfg['control']['settle_timeout_s']
    while time.monotonic()<deadline:
        motor.state()
        if imus.settled():
            break
        time.sleep(.1)
    else:
        raise DemoError('IMU settling timeout')
    motor.wait(cfg['control']['pause_s'])
    start=time.monotonic(); encoder=[]
    while time.monotonic()-start<cfg['control']['summary_s']:
        imus.ensure_alive()
        state=motor.state()
        encoder.append(motor.positions(state).tolist())
        time.sleep(.05)
    end=time.monotonic()
    features,quality=summarise(imus.snapshot(start,end),encoder,cfg)
    quality.update({'start_monotonic_s':start,'end_monotonic_s':end,'duration_s':end-start})
    return features,quality,(audit.summary(start,end) if audit else None)
