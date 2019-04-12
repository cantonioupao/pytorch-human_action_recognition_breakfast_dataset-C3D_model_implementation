import torch
import numpy as np
from torch import nn, optim
from network import C3D_model
import cv2
torch.backends.cudnn.benchmark = True

def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #use the second gpu
    print("Device being used:", device)

    with open('actions_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    #print(class_names)
    # init model
    model = C3D_model.C3D(num_classes=48)
    model = nn.DataParallel(model);
    checkpoint = torch.load('C3D39.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # read video
    video = 'P08_tea.avi'
    cap = cv2.VideoCapture(video)
    retaining = True

    clip = []
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            with torch.no_grad():
                outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)

            top5probs , labels = torch.topk(probs,5)
           

            

            #action 1 shown
            cv2.putText(frame, class_names[labels[0][0]].split(' ')[-1].strip(), (20, 20),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 5), 1,cv2.LINE_AA)
            cv2.putText(frame, "prob: %.4f" % top5probs[0][0], (110, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (30, 30, 255), 1, cv2.LINE_AA)



            #action 2 shown
            cv2.putText(frame, class_names[labels[0][1]].split(' ')[-1].strip(), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 5), 1,cv2.LINE_AA )
            cv2.putText(frame, "prob: %.4f" % top5probs[0][1], (110, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (30, 30, 255), 1, cv2.LINE_AA)



            #action 3 shown
            cv2.putText(frame, class_names[labels[0][2]].split(' ')[-1].strip(), (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 5), 1,cv2.LINE_AA)
            cv2.putText(frame, "prob: %.4f" % top5probs[0][2], (110, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (30, 30, 255), 1,cv2.LINE_AA)



            #action 4 shown
            cv2.putText(frame, class_names[labels[0][3]].split(' ')[-1].strip(), (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255,5 ), 1,cv2.LINE_AA)
            cv2.putText(frame, "prob: %.4f" % top5probs[0][3], (110, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (30, 30, 255), 1,cv2.LINE_AA)


            #action 5 shown
            cv2.putText(frame, class_names[labels[0][4]].split(' ')[-1].strip(), (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        ( 255,255,5 ), 1,cv2.LINE_AA)
            cv2.putText(frame, "prob: %.4f" % top5probs[0][4], (110, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (30, 30, 255), 1,cv2.LINE_AA)
            clip.pop(0)

        cv2.imshow('result', frame)
        cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()











