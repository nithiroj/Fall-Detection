import argparse
import cv2
import json

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--fname', required=True, 
    help='filename without extension for vdo.avi and info.json')

args = vars(ap.parse_args())

fname = args['fname']

vdo_path = f'{fname}.avi'
json_path = f'{fname}.json'

colors = {'Green': (0, 255, 0),
          'Red': (0, 0, 255)}

# Helpers
def get_infos(summary):
    infos = {}
    for info in summary:
        k = info.pop('id')
        v = info
        infos[k] = v
    return infos

def draw_box(frame, minX, minY, maxX, maxY, color, thickness=2):
    frame = cv2.rectangle(frame, (minX, minY), 
                        (maxX, maxY), color, thickness)
    return frame

with open(json_path) as f:
    summary = json.load(f)

infos = get_infos(summary)
cap = cv2.VideoCapture(vdo_path)

idx = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret : break
        
    idx += 1
        
    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    # If person detected, draw bounding box
    if infos.get(idx):
        info = infos[idx]
        if info.get('pos') == 'not_fall':
            frame = draw_box(frame, info['left'], info['top'], info['right'],
                             info['bottom'], colors['Green'])
        elif info.get('pos') == 'fall':
            frame = draw_box(frame, info['left'], info['top'], info['right'],
                             info['bottom'], colors['Red'])
            frame = cv2.putText(frame, '{}, prob: {:.2f}'.format('Fall', info['prob']),
                                (info['left'], info['top']-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['Red'], 2)
        else:
            frame = frame


    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()