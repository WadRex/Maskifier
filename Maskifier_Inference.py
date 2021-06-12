import cv2
import numpy as np
from yolov4.tf import YOLOv4

#yolo coco model 
yolo = YOLOv4()
yolo.config.parse_names("YOLO/yolov4_coco.names")
yolo.config.parse_cfg("YOLO/yolov4_coco.cfg")
yolo.make_model()
yolo.load_weights("YOLO/yolov4_coco.weights", weights_type="yolo")
yolo.summary(summary_type="yolo")
yolo.summary()

#yolo for head detection
headyolo = YOLOv4()
headyolo.config.parse_names("YOLO/yolov4_head.names")
headyolo.config.parse_cfg("YOLO/yolov4_head.cfg")
headyolo.make_model()
headyolo.load_weights("YOLO/yolov4_head.weights", weights_type="yolo")
headyolo.summary(summary_type="yolo")
headyolo.summary()

#get webcam
cam = cv2.VideoCapture(0)

#where everything happens
while True:
    #get readable video
    ret, frame = cam.read()
    
    #get human detection
    human_pred_bboxes = yolo.predict(frame, 0.5)             
    
    #preprocess human detection
    height, width, _ = frame.shape
    human_bboxes = human_pred_bboxes * np.array([width, height, width, height, 1, 1])
    for bbox in human_bboxes:
        #find human
        if int(bbox[4]) == 0:
            #check if probability not NULL
            if float(bbox[5]) > 0:
                #define coordinates
                y = int(bbox[1]) - int(bbox[3]) / int(2)
                x = int(bbox[0]) - int(bbox[2]) / int(2)
                h = int(bbox[3])
                w = int(bbox[2])
                
                #normalize?
                if y < 0:
                    y = 0
                if x < 0:
                    x = 0                                                        
                    
                #crop
                frame = frame[int(y):int(y+h), int(x):int(x+w)] 
                
                #get prediction if this human is wearing a mask
                head_pred_bboxes = headyolo.predict(frame, 0.5)
                frame = headyolo.draw_bboxes(frame, head_pred_bboxes)
                
                #preprocess mask detection
                height, width, _ = frame.shape
                head_bboxes = head_pred_bboxes * np.array([width, height, width, height, 1, 1])
                for head_bbox in head_bboxes:
                    #decide if mask is on or not    
                    if int(head_bbox[4]) == 1:
                        #check if probability not NULL
                        if float(head_bbox[5]) > 0:
                            cv2.rectangle(frame, (2,2), (135,35), (0,0,0), cv2.FILLED)
                            cv2.putText(frame, "Wearing Mask :)", (2,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1) 
                            cv2.putText(frame, "Accuracy: ~" + str(int(head_bbox[5] * 100)) + "%", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1) 
                                                        
                    if int(head_bbox[4]) == 0:
                        #check if probability not NULL
                        if float(head_bbox[5]) > 0:
                            cv2.rectangle(frame, (2,2), (165,35), (0,0,0), cv2.FILLED)
                            cv2.putText(frame, "Not Wearing Mask :(", (2,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                            cv2.putText(frame, "Accuracy: ~" + str(int(head_bbox[5] * 100)) + "%", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1) 
                    
                    if float(head_bbox[5]) == 0:
                        cv2.rectangle(frame, (2,2), (120,20), (0,0,0), cv2.FILLED)
                        cv2.putText(frame, "No Detection...", (2,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1) 
        
                
                #show results
                cv2.imshow("Maskifier", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break   
                
#free memory
cam.release()
cv2.destroyAllWindows()
yolo.close_session()

