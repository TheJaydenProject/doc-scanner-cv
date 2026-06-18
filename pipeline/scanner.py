import cv2
import numpy as np


class ContourNotFoundError(Exception):
    """Raised when no valid four-point document contour is found."""
    pass


def preprocess(image_bytes: bytes) -> np.ndarray:
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ContourNotFoundError("Could not decode image bytes.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (7, 7), 0)


def find_document_contour(blurred: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if not contours:
        raise ContourNotFoundError("No contours found in image.")

    # Pass 1: largest clean 4-point quad.
    for contour in contours[:8]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx

    # Pass 2: convex hull of the largest contour.
    hull = cv2.convexHull(contours[0])
    hull_approx = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
    if len(hull_approx) == 4:
        return hull_approx

    # Pass 3: minAreaRect over all edge pixels — works when the boundary
    # never forms a closed contour (e.g. low-contrast paper against similar background).
    edge_points = cv2.findNonZero(edges)
    if edge_points is not None:
        rect = cv2.minAreaRect(edge_points)
        box = cv2.boxPoints(rect)
        return np.intp(box).reshape(4, 1, 2)

    raise ContourNotFoundError("No four-point document contour found.")


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders corner points as [top-left, top-right, bottom-right, bottom-left].

    Uses per-point sum and diff rather than np.diff(axis=1), which produces
    wrong results for contours with non-standard winding order.
    """
    rect = np.zeros((4, 2), dtype="float32")
    pts = pts.reshape(4, 2).astype("float32")

    point_sums = pts[:, 0] + pts[:, 1]
    rect[0] = pts[np.argmin(point_sums)]   # top-left: smallest x+y
    rect[2] = pts[np.argmax(point_sums)]   # bottom-right: largest x+y

    point_diffs = pts[:, 0] - pts[:, 1]
    rect[1] = pts[np.argmax(point_diffs)]  # top-right: largest x-y
    rect[3] = pts[np.argmin(point_diffs)]  # bottom-left: smallest x-y

    return rect


def perspective_transform(image_bytes: bytes, contour: np.ndarray) -> np.ndarray:
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    rect = order_points(contour)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def binarize(warped: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51, 15
    )


def run_pipeline(image_bytes: bytes) -> np.ndarray:
    """
    Runs the full scanner pipeline.
    Returns the binarized image. If no document boundary is detected,
    falls back to binarizing the full frame without perspective correction.
    Raises ContourNotFoundError only if the image bytes cannot be decoded.
    """
    blurred = preprocess(image_bytes)
    try:
        contour = find_document_contour(blurred)
        warped = perspective_transform(image_bytes, contour)
        return binarize(warped)
    except ContourNotFoundError:
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return binarize(image)
