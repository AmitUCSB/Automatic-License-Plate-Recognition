# src/utils.py
import cv2, csv, time, os
from pathlib import Path

def draw_fps(img, fps):
    txt = f"FPS: {fps:.1f}"
    cv2.putText(img, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255,255,255), 2, cv2.LINE_AA)

def annotate_text(img, x1, y1, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.7, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img,(x1,max(0,y1-th-10)),(x1+tw+10,y1),(0,0,0),-1)
    cv2.putText(img,text,(x1+5,y1-5),font,scale,(255,255,255),thickness)

def save_csv_row(csv_path, row):
    first = not Path(csv_path).exists()
    os.makedirs(Path(csv_path).parent, exist_ok=True)
    with open(csv_path,"a",newline="") as f:
        w=csv.writer(f)
        if first:
            w.writerow(["timestamp","source","frame","text","conf","x1","y1","x2","y2","crop"])
        w.writerow(row)

def timestamp():
    return int(time.time())
