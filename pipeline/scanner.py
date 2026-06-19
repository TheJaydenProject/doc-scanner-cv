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


# A contour ratio outside [MIN, MAX] means find_document_contour locked onto
# something other than a real document boundary: below MIN it's a stray text
# blob or noise artifact; above MAX it's just tracing the image frame itself
# (e.g. a flat digital screenshot with no physical boundary at all).
MIN_CONTOUR_AREA_RATIO = 0.15
MAX_CONTOUR_AREA_RATIO = 0.97


def _quad_area_ratio(contour: np.ndarray, image_shape: tuple) -> float:
    """Shoelace-formula area of the 4-point contour as a fraction of the full image area."""
    pts = contour.reshape(4, 2).astype(np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    quad_area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    image_area = image_shape[0] * image_shape[1]
    return quad_area / image_area


INSET_RATIO = 0.025


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
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    # Symmetric inset crop removes the binding/shadow sliver that approxPolyDP
    # leaves at the contour edge. Cropping post-warp (not shrinking dst pre-warp)
    # keeps the margin centered on all four sides.
    dx = int(max_width * INSET_RATIO)
    dy = int(max_height * INSET_RATIO)
    if dx > 0 and dy > 0 and (max_width - 2 * dx) > 0 and (max_height - 2 * dy) > 0:
        warped = warped[dy:max_height - dy, dx:max_width - dx]

    return warped


def binarize_printed(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def binarize_handwritten(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    blurred = cv2.GaussianBlur(l_channel, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51, 15
    )


def remove_horizontal_lines(binary: np.ndarray) -> np.ndarray:
    """
    Erases ruled/feint lines (notebook or index-card rules) from a binarized
    image before MSER/OCR see it. Targets only long, thin horizontal strokes —
    a kernel this wide never matches real text characters or word connectors.
    """
    inverted = cv2.bitwise_not(binary)
    kernel_width = max(binary.shape[1] // 15, 1)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    detected_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    return cv2.bitwise_or(binary, detected_lines)


def run_pipeline(image_bytes: bytes) -> tuple[np.ndarray, bool]:
    """
    Runs perspective correction only. Returns (image, warped):
    image is the clean, unbinarized BGR frame (warped, or raw if skipped);
    warped is False when no document boundary was found, or when the
    detected quad's area ratio falls outside [MIN_CONTOUR_AREA_RATIO,
    MAX_CONTOUR_AREA_RATIO] — too small means the contour locked onto a
    stray text blob or noise artifact rather than a document edge; too
    large means it just traced the image frame (e.g. a flat digital
    screenshot with no physical boundary). Either way, warping it would
    distort rather than correct the image.
    Callers must classify_document() on this result, then
    pick binarize_printed() or binarize_handwritten() before OCR.
    Raises ContourNotFoundError only if the image bytes cannot be decoded.
    """
    blurred = preprocess(image_bytes)
    try:
        contour = find_document_contour(blurred)
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        ratio = _quad_area_ratio(contour, image.shape[:2])
        if ratio < MIN_CONTOUR_AREA_RATIO or ratio > MAX_CONTOUR_AREA_RATIO:
            return image, False
        return perspective_transform(image_bytes, contour), True
    except ContourNotFoundError:
        np_array = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(np_array, cv2.IMREAD_COLOR), False
