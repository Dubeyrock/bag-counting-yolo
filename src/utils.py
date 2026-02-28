import cv2

def draw_boxes(frame,boxes, track_ids, class_names,colors):
    '''draw bounding boxes and id on the frame.'''
    for box, tid, cls in zip(boxes, track_ids,class_names):
        x1,y1,x2,y2 = map(int,box[:4])
        label = f"{cls} ID:{tid}"
        color = colors.get(tid,(0,255,0))

        # Draw rectangle..
        cv2.rectangle(frame,(x1,y1), (x2,y2), color,2)
        # put label above box
        cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
    
    return frame
