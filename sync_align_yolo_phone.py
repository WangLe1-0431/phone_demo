"""
Author: WangLe1_
Email: leiw5385@gmail.com
Date: 2025-09-25
"""

import os, sys, platform, ctypes, argparse, time
from pathlib import Path

SDK_ROOT = r"D:\Desktop\pyorbbecsd\install" #改成你的电脑上对应的地址
DLL_DIRS = [
    os.path.join(SDK_ROOT, "lib"),
    os.path.join(SDK_ROOT, "bin"),
    os.path.join(SDK_ROOT, "extensions"),
]
for d in DLL_DIRS:
    if os.path.isdir(d):
        try:
            os.add_dll_directory(d)
        except Exception:
            pass
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

core_dll = os.path.join(SDK_ROOT, "lib", "OrbbecSDK.dll")
print("Loading:", core_dll)
ctypes.WinDLL(core_dll)

import cv2
import numpy as np
from ultralytics import YOLO

from pyorbbecsdk import *
from utils import frame_to_bgr_image

ESC_KEY = 27
MIN_DEPTH_MM = 20      # 20 mm
MAX_DEPTH_MM = 10000   # 10 m

def draw_fps(img, fps_val):
    cv2.putText(img, f"FPS: {fps_val:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

def build_point_cloud_xyzrgb(depth_mm, rgb, fx, fy, cx, cy, mask=None, depth_scale=0.001):
    """
    depth_mm: HxW uint16 (millimeters)
    rgb:      HxW x 3 uint8 (BGR)
    fx,fy,cx,cy: intrinsics
    mask:     HxW bool (optional), True to keep
    depth_scale: convert mm->m (default 0.001)
    Returns Nx3 XYZ (meters), Nx3 RGB (0-255 uint8)
    """
    H, W = depth_mm.shape
    z = depth_mm.astype(np.float32) * depth_scale
    if mask is None:
        valid = z > 0
    else:
        valid = (z > 0) & (mask.astype(bool))

    if not np.any(valid):
        return np.empty((0,3), np.float32), np.empty((0,3), np.uint8)

    vs, us = np.nonzero(valid)
    z_valid = z[vs, us]


    x = (us - cx) * z_valid / fx
    y = (vs - cy) * z_valid / fy
    XYZ = np.stack([x, y, z_valid], axis=1).astype(np.float32)

    # BGR -> RGB
    rgb_pts = rgb[vs, us, ::-1].copy()  # now RGB
    return XYZ, rgb_pts

def save_ply_xyzrgb(path, xyz, rgb):
    """
    path: file path string
    xyz: Nx3 float32 (meters)
    rgb: Nx3 uint8
    """
    n = xyz.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

def get_rgb_intrinsics_from_params(cam_param):

    intr = None
    if hasattr(cam_param, "rgb_intrinsic"):
        intr = cam_param.rgb_intrinsic
    elif hasattr(cam_param, "color_intrinsic"):
        intr = cam_param.color_intrinsic
    elif hasattr(cam_param, "depth_intrinsic"):
        intr = cam_param.depth_intrinsic
    else:
        raise RuntimeError("Cannot find intrinsics in camera_param")

    for k in ["fx","fy","cx","cy","width","height"]:
        if not hasattr(intr, k):
            raise RuntimeError("Intrinsic missing attribute: " + k)
    return intr.fx, intr.fy, intr.cx, intr.cy, intr.width, intr.height

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8s-seg.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    parser.add_argument("--enable_sync", type=bool, default=True)
    parser.add_argument("--align_to", default="color", choices=["color","depth"])
    parser.add_argument("--save_dir", default="ply_out")
    args = parser.parse_args(argv)

    import torch
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] Requested CUDA but no GPU visible. Falling back to CPU.")
            device = "cpu"

    yolo = YOLO(args.model)
    print(f"[INFO] YOLO: {args.model}, device={device}")

    # Orbbec pipeline
    pipeline = Pipeline()
    config = Config()
    try:
        plist = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_prof = plist.get_default_video_stream_profile()
        config.enable_stream(color_prof)
        plist = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_prof = plist.get_default_video_stream_profile()
        config.enable_stream(depth_prof)
    except Exception as e:
        print("[ERR] enabling streams:", e)
        return

    if args.enable_sync:
        try:
            pipeline.enable_frame_sync()
        except Exception as e:
            print("[WARN] enable_frame_sync:", e)

    try:
        pipeline.start(config)
    except Exception as e:
        print("[ERR] pipeline.start:", e)
        return

    # Align filter
    align_to = OBStreamType.COLOR_STREAM if args.align_to=="color" else OBStreamType.DEPTH_STREAM
    align_filter = AlignFilter(align_to_stream=align_to)

    try:
        cam_param = pipeline.get_camera_param()
        fx, fy, cx, cy, iw, ih = get_rgb_intrinsics_from_params(cam_param)
        print(f"[INFO] Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}, size={iw}x{ih}")
    except Exception as e:
        print("[WARN] cannot read intrinsics, will try from frames later:", e)
        fx=fy=cx=cy=None

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    t0, fcnt, fps_show = time.time(), 0, 0.0
    last_xyz, last_rgb = None, None

    win_name = "SyncAlign + YOLOv8 (phone mask)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    phone_names = {"cell phone"}

    print("按“P”保存点云到本地")
    print("按“Q”退出该程序")
    while True:
        try:
            frames = pipeline.wait_for_frames(100)
            if not frames: 
                continue

            # Align
            frames = align_filter.process(frames)
            if not frames: 
                continue
            frames = frames.as_frame_set()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # BGR color
            color_bgr = frame_to_bgr_image(color_frame)
            if color_bgr is None:
                continue


            try:
                d16 = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                    (depth_frame.get_height(), depth_frame.get_width()))
            except ValueError:
                continue
            scale = depth_frame.get_depth_scale()
            d16 = np.where((d16 >= MIN_DEPTH_MM) & (d16 <= MAX_DEPTH_MM), d16, 0).astype(np.uint16)


            res = yolo(color_bgr, imgsz=640, conf=args.conf, device=device, verbose=False)
            r0 = res[0]
            phone_mask = None
            if r0.masks is not None and len(r0.masks.data) > 0:
                # filter masks by class name
                names = r0.names if hasattr(r0, "names") else yolo.model.names
                clses = r0.boxes.cls.cpu().numpy().astype(int) if r0.boxes is not None else []
                keep = []
                for i, c in enumerate(clses):
                    cname = names.get(c, str(c)) if isinstance(names, dict) else (names[c] if c < len(names) else str(c))
                    if cname in phone_names:
                        keep.append(i)
                if keep:
                    # Combine kept masks
                    mk = r0.masks.data[keep].cpu().numpy()   # K x Hm x Wm (float 0/1)
                    mk = (mk.sum(axis=0) > 0.5).astype(np.uint8)  # Hm x Wm
                    # Resize to depth size (aligned to color -> same size usually, but be safe)
                    if mk.shape[:2] != d16.shape[:2]:
                        mk = cv2.resize(mk, (d16.shape[1], d16.shape[0]), interpolation=cv2.INTER_NEAREST)
                    phone_mask = mk.astype(bool)

            # ---------------- Visualization ----------------
            # depth colorized (for display only)
            depth_vis = cv2.normalize(d16, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(color_bgr, 0.55, depth_vis, 0.45, 0)

            if phone_mask is not None:
                vis = overlay.copy()
                # draw mask in pink
                pink = np.array([203, 0, 203], dtype=np.uint8)
                vis[phone_mask] = (0.5*vis[phone_mask] + 0.5*pink).astype(np.uint8)
            else:
                vis = overlay

            # fps
            fcnt += 1
            if fcnt >= 10:
                t1 = time.time(); fps_show = fcnt/(t1-t0+1e-9); t0=t1; fcnt=0
            draw_fps(vis, fps_show)

            cv2.imshow(win_name, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ESC_KEY):
                break

            # Build point cloud for export when 'p' pressed (or keep latest)
            if phone_mask is not None:
                # intrinsics: if not available yet, derive from color frame
                if fx is None:
                    # Try reading from the color stream profile (fallback)
                    try:
                        prof = color_frame.get_profile().as_video_stream_profile()
                        intr = prof.get_intrinsic()
                        fx, fy, cx, cy = intr.fx, intr.fy, intr.cx, intr.cy
                        print(f"[INFO] Intrinsics(from frame): fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
                    except Exception as e:
                        print("[WARN] ERROR", e)

                if fx is not None:
                    xyz, rgb = build_point_cloud_xyzrgb(
                        d16, color_bgr, fx, fy, cx, cy, mask=phone_mask, depth_scale=scale
                    )
                    if xyz.shape[0] > 0:
                        last_xyz, last_rgb = xyz, rgb

            if key == ord('p'):
                if last_xyz is None or last_xyz.shape[0] == 0:
                    print("[WARN] 未找到目标！")
                else:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    out_path = os.path.join(args.save_dir, f"phone_{ts}.ply")
                    save_ply_xyzrgb(out_path, last_xyz, last_rgb)
                    print(f"[OK] 已保存到: {out_path}  (点云中的点数为={last_xyz.shape[0]})")

        except KeyboardInterrupt:
            break
        except Exception as e:

            print("[WARN] loop exception:", e)

    cv2.destroyAllWindows()
    pipeline.stop()

if __name__ == "__main__":
    main(sys.argv[1:])
