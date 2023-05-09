import torch
import numpy as np
import cv2
from model_edge import Edge


def to_tensor(img):
    img = np.expand_dims(img, axis=2)
    img_tensor = torch.from_numpy(img.astype(np.float32))
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    return img_tensor


def to_np(img_tensor):
    img = img_tensor.cpu().permute(0, 2, 3, 1).numpy()[0]
    return img


def main():
    edge = Edge()
    img = cv2.imread('img/ny.png', cv2.IMREAD_GRAYSCALE)

    img_tensor = to_tensor(img)
    edge_tensor = edge(img_tensor)
    img_edges = to_np(edge_tensor)

    hor_grad_img = (255*img_edges[:, :, 0] / np.max(img_edges[:, :, 1])).astype(np.uint8)
    vert_grad_img = (255*img_edges[:, :, 1] / np.max(img_edges[:, :, 1])).astype(np.uint8)

    cv2.imshow('horizontal grad', hor_grad_img)
    cv2.imshow('vertical grad', vert_grad_img)
    cv2.waitKey()

    cv2.imwrite('img/ny_hor.png', hor_grad_img)
    cv2.imwrite('img/ny_vert.png', vert_grad_img)


if __name__ == "__main__":
    main()
