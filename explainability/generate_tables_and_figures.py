import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'..')
from feature_extraction.process_images import get_crop_image, calc_selected_distances
from deepface import DeepFace
import seaborn as sns
import matplotlib.pyplot as plt

def make_confusion_matrix(cf,ax,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Parameters
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False

    sns.heatmap(cf,ax=ax, annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)
    return ax

def get_computation_time_models(path_to_file):
    """
    Get computational time from different facial feature extraction techniques, in seconds per photograph
    
    Parameters
    ----------
    path_to_file : numpy array/pandas series
        Pandas series/numpy array containing all file paths to images to include in this analysis
        
    Returns
    -------
    mediapipe_time: float
        Seconds it takes mediapipe to process one photograph, mean overall whole dataset
    vgg_face_time: float
        Seconds it takes VGG-Face to process one photograph, mean overall whole dataset
    hybrid_time: float
        Seconds it takes the hybrid model to process one photograph, mean overall whole dataset
    mp_processed: float
        Ratio (between 0 - 1) of succesfully processed facial photographs for mediapipe
    vgg_processed: float
        Ratio (between 0 - 1) of succesfully processed facial photographs for VGG-Face
    hybrid_processed: float
        Ratio (between 0 - 1) of succesfully processed facial photographs for the hybrid model
    """
    import time
    import traceback
    hybrid_time = 158.901 #since it is in a VM, cannot calculate it dynamically
    hybrid_processed = 0.863
    
    EDGES = [(173, 155), (155, 133), (133, 173), (246, 33), (33, 7), (7, 246), (382, 398), (398, 362), (362, 382), (263, 466), (466, 249), (249, 263), (308, 415), (415, 324), (324, 308), (78, 95), (95, 191), (191, 78), (356, 389), (389, 264), (264, 356), (127, 34), (34, 162), (162, 127), (368, 264), (389, 368), (139, 162), (34, 139), (267, 0), (0, 302), (302, 267), (37, 72), (72, 0), (0, 37), (11, 302), (0, 11), (72, 11), (349, 451), (451, 350), (350, 349), (120, 121), (121, 231), (231, 120), (452, 350), (451, 452), (232, 231), (121, 232), (302, 269), (269, 267), (37, 39), (39, 72), (303, 269), (302, 303), (73, 72), (39, 73), (357, 343), (343, 350), (350, 357), (128, 121), (121, 114), (114, 128), (277, 350), (343, 277), (47, 114), (121, 47), (452, 357), (128, 232), (453, 357), (452, 453), (233, 232), (128, 233), (299, 333), (333, 297), (297, 299), (69, 67), (67, 104), (104, 69), (332, 297), (333, 332), (103, 104), (67, 103), (175, 152), (152, 396), (396, 175), (175, 171), (171, 152), (377, 396), (152, 377), (148, 152), (171, 148), (381, 384), (384, 382), (382, 381), (154, 155), (155, 157), (157, 154), (384, 398), (173, 157), (280, 347), (347, 330), (330, 280), (50, 101), (101, 118), (118, 50), (348, 330), (347, 348), (119, 118), (101, 119), (303, 270), (270, 269), (39, 40), (40, 73), (304, 270), (303, 304), (74, 73), (40, 74), (9, 336), (336, 151), (151, 9), (151, 107), (107, 9), (337, 151), (336, 337), (108, 107), (151, 108), (344, 278), (278, 360), (360, 344), (115, 131), (131, 48), (48, 115), (279, 360), (278, 279), (49, 48), (131, 49), (262, 431), (431, 418), (418, 262), (32, 194), (194, 211), (211, 32), (424, 418), (431, 424), (204, 211), (194, 204), (304, 408), (408, 270), (40, 184), (184, 74), (409, 270), (408, 409), (185, 184), (40, 185), (272, 310), (310, 407), (407, 272), (42, 183), (183, 80), (80, 42), (415, 407), (310, 415), (191, 80), (183, 191), (322, 270), (270, 410), (410, 322), (92, 186), (186, 40), (40, 92), (409, 410), (186, 185), (347, 449), (449, 348), (119, 229), (229, 118), (450, 348), (449, 450), (230, 229), (119, 230), (434, 432), (432, 430), (430, 434), (214, 210), (210, 212), (212, 214), (422, 430), (432, 422), (202, 212), (210, 202), (313, 314), (314, 18), (18, 313), (83, 18), (18, 84), (84, 83), (17, 18), (314, 17), (17, 84), (307, 375), (375, 306), (306, 307), (77, 76), (76, 146), (146, 77), (291, 306), (375, 291), (61, 146), (76, 61), (259, 387), (387, 260), (260, 259), (29, 30), (30, 160), (160, 29), (388, 260), (387, 388), (161, 160), (30, 161), (286, 414), (414, 384), (384, 286), (56, 157), (157, 190), (190, 56), (414, 398), (173, 190), (424, 406), (406, 418), (194, 182), (182, 204), (335, 406), (424, 335), (106, 204), (182, 106), (367, 416), (416, 364), (364, 367), (138, 135), (135, 192), (192, 138), (434, 364), (416, 434), (214, 192), (135, 214), (391, 423), (423, 327), (327, 391), (165, 98), (98, 203), (203, 165), (358, 327), (423, 358), (129, 203), (98, 129), (298, 301), (301, 284), (284, 298), (68, 54), (54, 71), (71, 68), (251, 284), (301, 251), (21, 71), (54, 21), (4, 275), (275, 5), (5, 4), (5, 45), (45, 4), (281, 5), (275, 281), (51, 45), (5, 51), (254, 373), (373, 253), (253, 254), (24, 23), (23, 144), (144, 24), (374, 253), (373, 374), (145, 144), (23, 145), (320, 321), (321, 307), (307, 320), (90, 77), (77, 91), (91, 90), (321, 375), (146, 91), (280, 425), (425, 411), (411, 280), (50, 187), (187, 205), (205, 50), (427, 411), (425, 427), (207, 205), (187, 207), (421, 313), (313, 200), (200, 421), (201, 200), (200, 83), (83, 201), (18, 200), (335, 321), (321, 406), (182, 91), (91, 106), (405, 406), (321, 405), (181, 91), (182, 181), (321, 404), (404, 405), (181, 180), (180, 91), (320, 404), (180, 90), (314, 16), (16, 17), (16, 84), (315, 16), (314, 315), (85, 84), (16, 85), (425, 266), (266, 426), (426, 425), (205, 206), (206, 36), (36, 205), (423, 426), (266, 423), (203, 36), (206, 203), (369, 396), (396, 400), (400, 369), (140, 176), (176, 171), (171, 140), (377, 400), (176, 148), (391, 269), (269, 322), (322, 391), (165, 92), (92, 39), (39, 165), (417, 465), (465, 413), (413, 417), (193, 189), (189, 245), (245, 193), (464, 413), (465, 464), (244, 245), (189, 244), (257, 258), (258, 386), (386, 257), (27, 159), (159, 28), (28, 27), (385, 386), (258, 385), (158, 28), (159, 158), (388, 467), (467, 260), (30, 247), (247, 161), (466, 467), (388, 466), (246, 161), (247, 246), (248, 456), (456, 419), (419, 248), (3, 196), (196, 236), (236, 3), (399, 419), (456, 399), (174, 236), (196, 174), (333, 298), (298, 332), (103, 68), (68, 104), (284, 332), (103, 54), (285, 8), (8, 417), (417, 285), (55, 193), (193, 8), (8, 55), (168, 417), (8, 168), (193, 168), (340, 261), (261, 346), (346, 340), (111, 117), (117, 31), (31, 111), (448, 346), (261, 448), (228, 31), (117, 228), (417, 441), (441, 285), (55, 221), (221, 193), (413, 441), (221, 189), (327, 460), (460, 326), (326, 327), (98, 97), (97, 240), (240, 98), (328, 326), (460, 328), (99, 240), (97, 99), (277, 355), (355, 329), (329, 277), (47, 100), (100, 126), (126, 47), (371, 329), (355, 371), (142, 126), (100, 142), (309, 392), (392, 438), (438, 309), (79, 218), (218, 166), (166, 79), (439, 438), (392, 439), (219, 166), (218, 219), (382, 256), (256, 381), (154, 26), (26, 155), (341, 256), (382, 341), (112, 155), (26, 112), (279, 420), (420, 360), (131, 198), (198, 49), (429, 420), (279, 429), (209, 49), (198, 209), (365, 364), (364, 379), (379, 365), (136, 150), (150, 135), (135, 136), (394, 379), (364, 394), (169, 135), (150, 169), (277, 437), (437, 355), (126, 217), (217, 47), (343, 437), (217, 114), (443, 444), (444, 282), (282, 443), (223, 52), (52, 224), (224, 223), (283, 282), (444, 283), (53, 224), (52, 53), (275, 363), (363, 281), (51, 134), (134, 45), (440, 363), (275, 440), (220, 45), (134, 220), (262, 395), (395, 431), (211, 170), (170, 32), (369, 395), (262, 369), (140, 32), (170, 140), (337, 299), (299, 338), (338, 337), (108, 109), (109, 69), (69, 108), (297, 338), (109, 67), (335, 273), (273, 321), (91, 43), (43, 106), (273, 375), (146, 43), (450, 349), (349, 348), (119, 120), (120, 230), (450, 451), (231, 230), (467, 359), (359, 342), (342, 467), (247, 113), (113, 130), (130, 247), (446, 342), (359, 446), (226, 130), (113, 226), (283, 334), (334, 282), (52, 105), (105, 53), (293, 334), (283, 293), (63, 53), (105, 63), (250, 458), (458, 462), (462, 250), (20, 242), (242, 238), (238, 20), (461, 462), (458, 461), (241, 238), (242, 241), (276, 353), (353, 300), (300, 276), (46, 70), (70, 124), (124, 46), (383, 300), (353, 383), (156, 124), (70, 156), (325, 292), (292, 324), (324, 325), (96, 95), (95, 62), (62, 96), (292, 308), (78, 62), (283, 276), (276, 293), (63, 46), (46, 53), (300, 293), (63, 70), (447, 264), (264, 345), (345, 447), (227, 116), (116, 34), (34, 227), (372, 345), (264, 372), (143, 34), (116, 143), (352, 345), (345, 346), (346, 352), (123, 117), (117, 116), (116, 123), (345, 340), (111, 116), (1, 19), (19, 274), (274, 1), (1, 44), (44, 19), (354, 274), (19, 354), (125, 19), (44, 125), (248, 281), (281, 456), (236, 51), (51, 3), (363, 456), (236, 134), (426, 427), (207, 206), (436, 427), (426, 436), (216, 206), (207, 216), (380, 381), (381, 252), (252, 380), (153, 22), (22, 154), (154, 153), (256, 252), (22, 26), (391, 393), (393, 269), (39, 167), (167, 165), (393, 267), (37, 167), (199, 428), (428, 200), (200, 199), (200, 208), (208, 199), (428, 421), (201, 208), (330, 329), (329, 266), (266, 330), (101, 36), (36, 100), (100, 101), (371, 266), (36, 142), (432, 273), (273, 422), (202, 43), (43, 212), (287, 273), (432, 287), (57, 212), (43, 57), (290, 250), (250, 328), (328, 290), (60, 99), (99, 20), (20, 60), (462, 328), (99, 242), (258, 286), (286, 385), (158, 56), (56, 28), (384, 385), (158, 157), (446, 353), (353, 342), (113, 124), (124, 226), (265, 353), (446, 265), (35, 226), (124, 35), (386, 259), (259, 257), (27, 29), (29, 159), (386, 387), (160, 159), (422, 431), (431, 430), (210, 211), (211, 202), (422, 424), (204, 202), (445, 342), (342, 276), (276, 445), (225, 46), (46, 113), (113, 225), (422, 335), (106, 202), (306, 292), (292, 307), (77, 62), (62, 76), (325, 307), (77, 96), (366, 447), (447, 352), (352, 366), (137, 123), (123, 227), (227, 137), (302, 268), (268, 303), (73, 38), (38, 72), (271, 303), (268, 271), (41, 38), (73, 41), (371, 358), (358, 266), (36, 129), (129, 142), (327, 294), (294, 460), (240, 64), (64, 98), (455, 460), (294, 455), (235, 64), (240, 235), (294, 331), (331, 278), (278, 294), (64, 48), (48, 102), (102, 64), (331, 279), (49, 102), (271, 304), (74, 41), (272, 304), (271, 272), (42, 41), (74, 42), (436, 434), (434, 427), (207, 214), (214, 216), (436, 432), (212, 216), (272, 408), (184, 42), (407, 408), (184, 183), (394, 430), (430, 395), (395, 394), (169, 170), (170, 210), (210, 169), (369, 378), (378, 395), (170, 149), (149, 140), (400, 378), (149, 176), (296, 334), (334, 299), (299, 296), (66, 69), (69, 105), (105, 66), (334, 333), (104, 105), (168, 351), (351, 417), (193, 122), (122, 168), (6, 351), (168, 6), (122, 6), (411, 352), (352, 280), (50, 123), (123, 187), (376, 352), (411, 376), (147, 187), (123, 147), (319, 320), (320, 325), (325, 319), (89, 96), (96, 90), (90, 89), (285, 295), (295, 336), (336, 285), (55, 107), (107, 65), (65, 55), (296, 336), (295, 296), (66, 65), (107, 66), (320, 403), (403, 404), (180, 179), (179, 90), (319, 403), (179, 89), (348, 329), (100, 119), (349, 329), (100, 120), (293, 333), (104, 63), (293, 298), (68, 63), (323, 454), (454, 366), (366, 323), (93, 137), (137, 234), (234, 93), (454, 447), (227, 234), (315, 15), (15, 16), (15, 85), (316, 15), (315, 316), (86, 85), (15, 86), (279, 358), (358, 429), (209, 129), (129, 49), (331, 358), (129, 102), (316, 14), (14, 15), (14, 86), (317, 14), (316, 317), (87, 86), (14, 87), (285, 9), (9, 8), (9, 55), (349, 277), (47, 120), (252, 253), (253, 380), (153, 23), (23, 22), (374, 380), (153, 145), (402, 403), (403, 318), (318, 402), (178, 88), (88, 179), (179, 178), (319, 318), (88, 89), (6, 419), (419, 351), (122, 196), (196, 6), (197, 419), (6, 197), (196, 197), (324, 318), (318, 325), (96, 88), (88, 95), (397, 367), (367, 365), (365, 397), (172, 136), (136, 138), (138, 172), (288, 435), (435, 397), (397, 288), (58, 172), (172, 215), (215, 58), (435, 367), (138, 215), (439, 344), (344, 438), (218, 115), (115, 219), (439, 278), (48, 219), (271, 311), (311, 272), (42, 81), (81, 41), (311, 310), (80, 81), (281, 195), (195, 5), (195, 51), (248, 195), (195, 3), (287, 375), (146, 57), (287, 291), (61, 57), (396, 428), (428, 175), (175, 208), (208, 171), (199, 175), (268, 312), (312, 271), (41, 82), (82, 38), (312, 311), (81, 82), (444, 445), (445, 283), (53, 225), (225, 224), (254, 339), (339, 373), (144, 110), (110, 24), (390, 373), (339, 390), (163, 110), (144, 163), (295, 282), (282, 296), (66, 52), (52, 65), (448, 347), (347, 346), (117, 118), (118, 228), (448, 449), (229, 228), (454, 356), (356, 447), (227, 127), (127, 234), (296, 337), (108, 66), (337, 10), (10, 151), (10, 108), (338, 10), (10, 109), (439, 294), (64, 219), (439, 455), (235, 219), (415, 292), (292, 407), (183, 62), (62, 191), (371, 429), (209, 142), (355, 429), (209, 126), (372, 340), (111, 143), (265, 340), (372, 265), (35, 143), (111, 35), (388, 390), (390, 466), (246, 163), (163, 161), (390, 249), (7, 163), (346, 280), (50, 117), (295, 442), (442, 282), (52, 222), (222, 65), (442, 443), (223, 222), (19, 94), (94, 354), (125, 94), (370, 354), (94, 370), (141, 94), (125, 141), (285, 442), (222, 55), (441, 442), (222, 221), (197, 248), (3, 197), (197, 195), (359, 263), (263, 255), (255, 359), (130, 25), (25, 33), (33, 130), (249, 255), (25, 7), (275, 274), (274, 440), (220, 44), (44, 45), (457, 440), (274, 457), (237, 44), (220, 237), (383, 301), (301, 300), (70, 71), (71, 156), (368, 301), (383, 368), (139, 156), (71, 139), (351, 465), (245, 122), (412, 465), (351, 412), (188, 122), (245, 188), (263, 467), (247, 33), (389, 251), (251, 368), (139, 21), (21, 162), (374, 386), (386, 380), (153, 159), (159, 145), (385, 380), (153, 158), (394, 378), (378, 379), (150, 149), (149, 169), (419, 412), (188, 196), (399, 412), (188, 174), (426, 322), (322, 436), (216, 92), (92, 206), (410, 436), (216, 186), (387, 373), (373, 388), (161, 144), (144, 160), (393, 326), (326, 164), (164, 393), (167, 164), (164, 97), (97, 167), (2, 164), (326, 2), (2, 97), (370, 461), (461, 354), (125, 241), (241, 141), (370, 462), (242, 141), (267, 164), (164, 0), (164, 37), (11, 12), (12, 302), (72, 12), (12, 268), (38, 12), (374, 387), (160, 145), (12, 13), (13, 268), (38, 13), (13, 312), (82, 13), (300, 298), (68, 70), (265, 261), (31, 35), (446, 261), (31, 226), (385, 381), (154, 158), (330, 425), (205, 101), (391, 426), (206, 165), (355, 420), (198, 126), (437, 420), (198, 217), (327, 393), (167, 98), (457, 438), (438, 440), (220, 218), (218, 237), (344, 440), (220, 115), (362, 341), (112, 133), (463, 341), (362, 463), (243, 133), (112, 243), (457, 461), (461, 459), (459, 457), (237, 239), (239, 241), (241, 237), (458, 459), (239, 238), (430, 364), (135, 210), (414, 463), (463, 398), (173, 243), (243, 190), (262, 428), (428, 369), (140, 208), (208, 32), (274, 461), (241, 44), (316, 403), (403, 317), (87, 179), (179, 86), (402, 317), (87, 178), (315, 404), (404, 316), (86, 180), (180, 85), (314, 405), (405, 315), (85, 181), (181, 84), (313, 406), (406, 314), (84, 182), (182, 83), (406, 421), (421, 418), (194, 201), (201, 182), (366, 401), (401, 323), (93, 177), (177, 137), (361, 323), (401, 361), (132, 177), (93, 132), (407, 306), (306, 408), (184, 76), (76, 183), (306, 409), (185, 76), (291, 409), (185, 61), (409, 287), (287, 410), (186, 57), (57, 185), (410, 432), (212, 186), (416, 427), (207, 192), (416, 411), (187, 192), (368, 372), (143, 139), (383, 372), (143, 156), (459, 438), (218, 239), (459, 309), (79, 239), (376, 366), (137, 147), (376, 401), (177, 147), (4, 1), (1, 275), (45, 1), (262, 421), (201, 32), (358, 294), (64, 129), (435, 416), (192, 215), (433, 416), (435, 433), (213, 215), (192, 213), (439, 289), (289, 455), (235, 59), (59, 219), (392, 289), (59, 166), (462, 326), (97, 242), (370, 326), (97, 141), (370, 2), (2, 141), (94, 2), (455, 305), (305, 460), (240, 75), (75, 235), (289, 305), (75, 59), (448, 339), (339, 449), (229, 110), (110, 228), (254, 449), (229, 24), (446, 255), (255, 261), (31, 25), (25, 226), (254, 450), (230, 24), (253, 450), (230, 23), (253, 451), (231, 23), (252, 451), (231, 22), (252, 452), (232, 22), (256, 452), (232, 26), (341, 452), (232, 112), (341, 453), (233, 112), (464, 414), (414, 413), (189, 190), (190, 244), (464, 463), (243, 244), (413, 286), (286, 441), (221, 56), (56, 189), (286, 442), (222, 56), (258, 442), (222, 28), (258, 443), (223, 28), (257, 443), (223, 27), (443, 259), (259, 444), (224, 29), (29, 223), (260, 444), (224, 30), (260, 445), (225, 30), (467, 445), (225, 247), (250, 309), (309, 458), (238, 79), (79, 20), (290, 305), (305, 392), (392, 290), (60, 166), (166, 75), (75, 60), (305, 328), (99, 75), (376, 433), (433, 401), (177, 213), (213, 147), (435, 401), (177, 215), (290, 309), (79, 60), (416, 376), (147, 192), (463, 453), (233, 243), (464, 453), (233, 244), (464, 357), (128, 244), (465, 357), (128, 245), (412, 343), (343, 465), (245, 114), (114, 188), (343, 399), (399, 437), (217, 174), (174, 114), (440, 360), (360, 363), (134, 131), (131, 220), (456, 420), (420, 399), (174, 198), (198, 236), (363, 420), (198, 134), (401, 288), (288, 361), (132, 58), (58, 177), (265, 383), (156, 35), (249, 339), (339, 255), (25, 110), (110, 7), (255, 448), (228, 25), (317, 13), (13, 14), (13, 87), (317, 312), (82, 87), (402, 312), (82, 178), (402, 311), (81, 178), (318, 311), (81, 88), (318, 310), (80, 88), (324, 310), (80, 95)]
    
    start = time.time()
    
    failure_mp = 0
    
    for y,file in enumerate(path_to_file):
           if len(str(file)) < 4:
                continue
           try:
               r,z,t = get_crop_image(file)
               dist = calc_selected_distances(r, EDGES, 'euclidean')
           except Exception:
               traceback.print_exc()
               failure_mp += 1
               print(file)
               continue
    end = time.time()
    
    mediapipe_time = (end - start) / len(path_to_file)
    
    failure_vgg = 0
    
    start = time.time()
    for y,file in enumerate(path_to_file):
           if len(str(file)) < 4:
                continue
           try:
               dist = DeepFace.represent(file, detector_backend ='mtcnn')
           except Exception:
               traceback.print_exc()
               failure_vgg += 1
               print(file)
               continue
    end = time.time()
    
    vgg_face_time = (end - start) / len(path_to_file)
    
    vgg_processed = (len(path_to_file) - failure_vgg)/len(path_to_file)
    mp_processed = (len(path_to_file) - failure_mp)/len(path_to_file)
    return mediapipe_time, vgg_face_time, hybrid_time, mp_processed, vgg_processed, hybrid_processed

def confusion_matrices(df_results_20, df_results_40):
    """
    Get confusion matrices for the different analyses and save the figure
    
    Parameters
    ----------
    df_results : pandas dataframe 
        Dataframe with the results from the softmax regression (see softmax_regression.py)
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(20,18))
    
    axs = []
    gs = gridspec.GridSpec(1, 4)
    gs.update(wspace=0.5)
    gs.update(hspace=0.3)
    axs.append(fig.add_subplot(gs[0, :2], ))
    axs.append(fig.add_subplot(gs[0, 2:]))
    
    titles = ['Confusion matrix (n=20 per class)', 'Confusion matrix (n=40 per class)', 'Confusion matrix (matched dataset)']

    for i in range(2):
        if i == 0:
            df_results = df_results_20
        elif i == 1:
            df_results = df_results_40
        y_true = df_results.y_true_vgg.explode().to_numpy(dtype=int)
        y_pred = df_results.vgg_classes.explode().to_numpy(dtype=int)
        
        cm = confusion_matrix(y_true, y_pred)
        
        n_syndromes = ['DDX3X',
         'DYRK1A',
         'ADNP',
         'ANKRD11',
         'ARID1B',
         'CHD8',
         'FBXO11',
         'MED13L',
         'PACS1',
         'PHIP',
         'PPM1D',
         'PURA',
         'YY1',
         'SON',
         'EHMT1',
         'KDM3B',
         '22q11.2 Deletion',
         'KANSL1']
        
        make_confusion_matrix(cm,
                              ax=axs[i], figsize=(20,15), categories=n_syndromes, percent=False)
        axs[i].set_ylabel('True label')
        axs[i].set_xlabel('Predicted label')
        axs[i].set_title(titles[i])
   
    plt.savefig("fig_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.show()
    return

def print_results(df_results_20, df_results_40):
    """
    Generate the results table with the results
    
    Parameters
    ----------
    df_results : pandas dataframe 
        Dataframe with the results from the softmax regression (see softmax_regression.py)
    """
    from sklearn.metrics import log_loss
    import numpy as np, scipy.stats as st
    from scipy.stats import mannwhitneyu
  
    df_table = pd.DataFrame(columns=['Accuracy', 'Log loss'])
    
    titles = ['_20', '_40']
    
    for z, df_results in enumerate([df_results_20, df_results_40]):
        accuracies_all, log_loss_all = np.zeros((len(df_results), 3)), np.zeros((len(df_results),3))
        
        for y in range(3):
            accuracies = []
            log_losses = []
            for i in range(len(df_results)):
                if y == 0:
                    accuracies.append(np.mean(df_results.y_true_vgg[i] == df_results.vgg_classes[i]))
                    log_losses.append(log_loss(df_results.y_true_vgg[i], df_results.vgg_pred[i], labels=np.unique(df_results.y_true_mp[i])))
                    name = 'VGG Face' + titles[z]
                if y == 1:
                    accuracies.append(np.mean(df_results.y_true_mp[i] == df_results.mp_classes[i]))
                    log_losses.append(log_loss(df_results.y_true_mp[i], df_results.mp_pred[i], labels=np.unique(df_results.y_true_mp[i])))
                    name = 'MediaPipe' + titles[z]
                if y == 2:
                    accuracies.append(np.mean(df_results.y_true_hybrid[i] == df_results.hybrid_classes[i]))
                    log_losses.append(log_loss(df_results.y_true_hybrid[i], df_results.hybrid_pred[i], labels=np.unique(df_results.y_true_mp[i])))
                    name = 'Hybrid model' + titles[z]
            
            min_ci, max_ci = np.round(st.t.interval(0.95, len(accuracies)-1, loc=np.mean(accuracies), scale=st.sem(accuracies)),2)
            df_table.at[name, 'Accuracy'] = str(np.round(np.mean(accuracies),2)) + " [" + str(min_ci) + "-" + str(max_ci) + "]"
            
            min_ci, max_ci = np.round(st.t.interval(0.95, len(log_losses)-1, loc=np.mean(log_losses), scale=st.sem(log_losses)),2)
            df_table.at[name, 'Log loss'] = str(np.round(np.mean(log_losses),2)) + " [" + str(min_ci) + "-" + str(max_ci) + "]"
            
            accuracies_all[:,y], log_loss_all[:,y] = accuracies, log_losses
        
        U1, p = mannwhitneyu(accuracies_all[:,0], accuracies_all[:,1])
        print("P VGG Face vs MediaPipe: " + str(p))
        U1, p = mannwhitneyu(accuracies_all[:,1], accuracies_all[:,2])
        print("P Hybrid vs MediaPipe: " + str(p))
        U1, p = mannwhitneyu(accuracies_all[:,0], accuracies_all[:,2])
        print("P VGG Face vs Hybrid: " + str(p))
        
        U1, p = mannwhitneyu(log_loss_all[:,0], log_loss_all[:,1])
        print("P VGG Face vs MediaPipe: " + str(p))
        U1, p = mannwhitneyu(log_loss_all[:,1], log_loss_all[:,2])
        print("P Hybrid vs MediaPipe: " + str(p))
        U1, p = mannwhitneyu(log_loss_all[:,0], log_loss_all[:,2])
        print("P VGG Face vs Hybrid: " + str(p))

    # mp_time, vgg_time, hybrid_time, mp_processed, vgg_processed, hybrid_processed = get_computation_time_models(path_to_files) #if you want to calculate this, need path to images
    # df_table["Execution time (seconds per photo)"] = np.round([mp_time, vgg_time, hybrid_time],1)
    # df_table["Succesfully processed photos (%)"] = np.round([mp_processed*100, vgg_processed*100, hybrid_processed*100],1)
    df_table.to_excel("results_softmax.xlsx")
    df_table.to_latex("results_softmax.tex",bold_rows=True)
    return


