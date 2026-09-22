"""Probe SteamVR for connected tracker devices and read one pose sample,
to confirm real OpenVR access works before wiring it into the collector.
"""
import openvr

vr = openvr.init(openvr.VRApplication_Other)

print("=== enumerating devices ===")
tracker_index = None
for i in range(openvr.k_unMaxTrackedDeviceCount):
    cls = vr.getTrackedDeviceClass(i)
    if cls == openvr.TrackedDeviceClass_Invalid:
        continue
    class_name = {
        openvr.TrackedDeviceClass_HMD: "HMD",
        openvr.TrackedDeviceClass_Controller: "Controller",
        openvr.TrackedDeviceClass_GenericTracker: "GenericTracker",
        openvr.TrackedDeviceClass_TrackingReference: "TrackingReference(base station)",
    }.get(cls, f"Unknown({cls})")
    serial = vr.getStringTrackedDeviceProperty(i, openvr.Prop_SerialNumber_String)
    print(f"  device {i}: class={class_name} serial={serial}")
    if cls == openvr.TrackedDeviceClass_GenericTracker and tracker_index is None:
        tracker_index = i

if tracker_index is None:
    print("\nNo GenericTracker found.")
else:
    print(f"\n=== reading pose from tracker device {tracker_index} ===")
    poses = vr.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
    )
    p = poses[tracker_index]
    print(f"  bPoseIsValid={p.bPoseIsValid} bDeviceIsConnected={p.bDeviceIsConnected} "
          f"eTrackingResult={p.eTrackingResult}")
    m = p.mDeviceToAbsoluteTracking
    matrix = [[m[r][c] for c in range(4)] for r in range(3)]
    print(f"  3x4 matrix: {matrix}")
    print(f"  position (x,y,z): ({matrix[0][3]}, {matrix[1][3]}, {matrix[2][3]})")

openvr.shutdown()
print("=== done ===")
