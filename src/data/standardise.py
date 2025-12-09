import cv2

def standardise_image(img, resize=(224, 224), clahe=False, denoise=False):
    if img is None:
        return None

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