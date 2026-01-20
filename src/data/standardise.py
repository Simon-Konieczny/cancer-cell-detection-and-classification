import cv2
import numpy as np

def standardise_image(img, resize=(224, 224), clahe=False, denoise=False, remove_artifacts=True):
    if img is None:
        return None
    
    if remove_artifacts:
        _, mask = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
        
        # Clean mask of small noise/text
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Keep only the largest object
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Mask out background artifacts
            clean_mask = np.zeros_like(mask)
            cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            img = cv2.bitwise_and(img, img, mask=clean_mask)
            
            # Crop to the breast area
            x, y, w, h = cv2.boundingRect(largest_contour)
            img = img[y:y+h, x:x+w]
        
        # INPAINTING TO REMOVE ANNOTATIONS
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        
        _, annot_mask = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)
        
        annot_mask = cv2.medianBlur(annot_mask, 3)
        
        img = cv2.inpaint(img, annot_mask, 3, cv2.INPAINT_TELEA)

    if denoise:
        img = cv2.medianBlur(img, 3)

    if clahe:
        clahe_fn = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe_fn.apply(img)

    if resize:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)

    h = dhash_cv(img)

    return (img, h)

def apply_standardization(img_path, out_path, resize=(224, 224), clahe=False, denoise=False):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    result = standardise_image(img, resize=resize, clahe=clahe, denoise=denoise)
    if result is None:
        return None
    img, h = result
    cv2.imwrite(str(out_path), img)
    return h

def dhash_cv(img, hash_size=8):
    # convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize to (hash_size+1, hash_size)
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)

    diff = resized[:, 1:] > resized[:, :-1]

    return int(''.join(diff.flatten().astype(int).astype(str)), 2)