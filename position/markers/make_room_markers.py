"""Print-ready A4 sheets for the room markers: 2 adjacent walls x (lower, upper) x (left, right) = 8 markers, DICT_4X4_50.

Wall A: lower 10 11, upper 14 15. Wall B: lower 12 13, upper 16 17 (id 7 stays the hand-held one).
Print at 100% / actual size; the black square must measure MARKER_MM, check with the 100 mm bar.

The setup that was flown uses one wall with IDs 10-15 only (10/12/14 lower row, 11/13/15 upper row, positions as
measured in ../room_layout.json); the wall / row labels printed on the sheets are from an earlier two-wall plan.
Whatever you print, measure where the markers ended up and put that in room_layout.json.
"""
import cv2
import numpy as np
from PIL import Image

DPI = 300
MARKER_MM = 160
A4 = (2480, 3508)                                    # px at 300 dpi
px = lambda mm: int(round(mm / 25.4 * DPI))

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
side = px(MARKER_MM) // 6 * 6                        # 6 modules across, keep them whole pixels
pages = []
for row_i, row in enumerate(["lower", "upper"]):
    for wall_i, (wall, col) in enumerate([("A", "left"), ("A", "right"), ("B", "left"), ("B", "right")]):
        mid = 10 + 4 * row_i + wall_i
        page = np.full((A4[1], A4[0]), 255, np.uint8)
        x0, y0 = (A4[0] - side) // 2, px(45)
        page[y0:y0 + side, x0:x0 + side] = cv2.aruco.generateImageMarker(dictionary, mid, side)
        y = y0 + side + px(30)
        cv2.putText(page, "ID %d  wall %s  %s %s" % (mid, wall, row, col), (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.6, 0, 6, cv2.LINE_AA)
        cv2.putText(page, "DICT_4X4_50, %d mm black square, this edge up" % MARKER_MM, (x0, y + px(14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3, cv2.LINE_AA)
        cv2.putText(page, "^ TOP ^", (A4[0] // 2 - px(18), y0 - px(12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 4, cv2.LINE_AA)
        by = y + px(32)                              # 100 mm bar to check the print scale
        cv2.line(page, (x0, by), (x0 + px(100), by), 0, 6)
        for bx in (x0, x0 + px(100)):
            cv2.line(page, (bx, by - px(4)), (bx, by + px(4)), 0, 6)
        cv2.putText(page, "100 mm", (x0 + px(104), by + px(2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3, cv2.LINE_AA)
        name = "marker_id%02d_wall%s_%s_%s.png" % (mid, wall, row, col)
        cv2.imwrite(name, page)
        pages.append(Image.fromarray(page))
        print(name)
pages[0].save("room_markers_A4.pdf", save_all=True, append_images=pages[1:], resolution=DPI)
print("room_markers_A4.pdf  (marker side %d px = %.1f mm)" % (side, side / DPI * 25.4))
