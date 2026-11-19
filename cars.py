from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from random import shuffle



def read_rgb_image(path: str | Path) -> np.ndarray:
    """Read an 8-bit RGB photograph.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a JPEG or PNG file.

    Returns
    -------
    rgb : np.ndarray
        RGB image, shape (H, W, 3), dtype uint8, values in [0, 255].
    """
    rgb = iio.imread(path)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f'Expected an 8-bit RGB image at {path}; got {rgb.shape}/{rgb.dtype}.')
    return rgb



def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert 8-bit RGB to encoded grayscale intensities.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image, shape (H, W, 3), dtype uint8.

    Returns
    -------
    gray : np.ndarray
        Grayscale image, shape (H, W), dtype float64,
        values in [0, 1]. This is an encoded grayscale, not linear luminance.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    return gray


@dataclass
class RegistrationResult:
    """Outputs of a complete two-image registration pipeline.

    Attributes
    ----------
    keypoints1, keypoints2 : np.ndarray
        Keypoints, shapes (N, 3) and (M, 3); rows are (x, y, sigma).
    descriptors1, descriptors2 : np.ndarray
        Descriptor matrices, shapes (N, D) and (M, D).
    matches : np.ndarray
        Accepted index pairs (index_in_image1, index_in_image2),
        shape (K, 2), dtype int64.
    homography : np.ndarray or None
        Projective map from image 1 to image 2, shape (3, 3), or None.
    inliers : np.ndarray
        Boolean RANSAC inlier mask, shape (K,).
    """
    keypoints1: np.ndarray
    keypoints2: np.ndarray
    descriptors1: np.ndarray
    descriptors2: np.ndarray
    matches: np.ndarray
    homography: np.ndarray | None
    inliers: np.ndarray


def reference_sift_features(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect and describe features using OpenCV's full SIFT.

    Parameters
    ----------
    gray : np.ndarray
        Encoded grayscale image, shape (H, W), float in [0, 1].

    Returns
    -------
    keypoints : np.ndarray
        Shape (N, 3), float64; columns are x, y and approximate sigma
        in image pixels (OpenCV's keypoint.size / 2).
    descriptors : np.ndarray
        Shape (N, 128), float64. These are OpenCV SIFT descriptors.
    """
    gray_u8 = np.uint8(np.round(np.clip(gray, 0, 1) * 255))
    sift = cv2.SIFT_create(nfeatures=1600) # type: ignore
    kp_cv, desc_cv = sift.detectAndCompute(gray_u8, None)
    if not kp_cv:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 128), dtype=np.float64)
    keypoints = np.array([(p.pt[0], p.pt[1], p.size / 2.0) for p in kp_cv], dtype=np.float64)
    descriptors = np.asarray(desc_cv, dtype=np.float64)
    return keypoints, descriptors


def reference_match_descriptors(desc1: np.ndarray, desc2: np.ndarray,
                                ratio: float = 0.78) -> np.ndarray:
    """Match float descriptors with OpenCV's 2-neighbor ratio test.

    Parameters
    ----------
    desc1, desc2 : np.ndarray
        Descriptor matrices, shapes (N, D) and (M, D), float.
        Both must have the same descriptor dimension D.
    ratio : float, optional
        Maximum accepted nearest/second-nearest distance ratio.

    Returns
    -------
    matches : np.ndarray
        Index pairs, shape (K, 2), dtype int64, K >= 0.
    """
    if len(desc1) == 0 or len(desc2) < 2:
        return np.empty((0, 2), dtype=np.int64)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    neighbors = bf.knnMatch(np.asarray(desc1, np.float32),
                            np.asarray(desc2, np.float32), k=2)
    pairs = [(a.queryIdx, a.trainIdx) for ns in neighbors if len(ns) == 2
             for a, b in [ns] if a.distance < ratio * b.distance]
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def estimate_homography(keypoints1: np.ndarray, keypoints2: np.ndarray,
                        matches: np.ndarray, threshold: float = 4.0
                        ) -> tuple[np.ndarray | None, np.ndarray]:
    """Fit a homography from matched points using library RANSAC.

    Parameters
    ----------
    keypoints1, keypoints2 : np.ndarray
        Keypoints from each image, shapes (N, 3) and (M, 3).
        Each row is (x, y, sigma); only x and y are used here.
    matches : np.ndarray
        Index pairs, shape (K, 2), dtype int.
    threshold : float, optional
        RANSAC reprojection threshold in pixels, default 4.0.

    Returns
    -------
    homography : np.ndarray or None
        Shape (3, 3), mapping image 1 coordinates into image 2,
        or None if there is insufficient geometric support.
    inliers : np.ndarray
        Boolean array of shape (K,), True for RANSAC inliers.
    """
    inliers = np.zeros(len(matches), dtype=bool)
    if len(matches) < 4:
        return None, inliers
    pts1 = np.float32(keypoints1[matches[:, 0], :2])
    pts2 = np.float32(keypoints2[matches[:, 1], :2])
    homography, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, # type: ignore
                                        ransacReprojThreshold=threshold,
                                        maxIters=3000, confidence=0.995) # type: ignore
    if homography is not None and mask is not None:
        inliers = mask.ravel().astype(bool)
    return homography, inliers


def reference_sift_pipeline(gray1: np.ndarray, gray2: np.ndarray,
                            ratio: float = 0.78) -> RegistrationResult:
    """Run a complete reference SIFT → matches → RANSAC pipeline.

    Parameters
    ----------
    gray1, gray2 : np.ndarray
        Encoded grayscale images, shapes (H1, W1) and (H2, W2),
        float in [0,1]. Images need not have identical dimensions.
    ratio : float, optional
        Lowe nearest-neighbor ratio threshold.

    Returns
    -------
    result : RegistrationResult
        Keypoints (N,3)/(M,3), descriptors (N,128)/(M,128),
        matches (K,2), homography (3,3) or None, inliers (K,).
    """
    kp1, d1 = reference_sift_features(gray1)
    kp2, d2 = reference_sift_features(gray2)
    matches = reference_match_descriptors(d1, d2, ratio=ratio)
    H, inliers = estimate_homography(kp1, kp2, matches)
    return RegistrationResult(kp1, kp2, d1, d2, matches, H, inliers)
def plot_matches(rgb1: np.ndarray, rgb2: np.ndarray,
                 keypoints1: np.ndarray, keypoints2: np.ndarray,
                 matches: np.ndarray, inliers: np.ndarray | None = None,
                 max_lines: int = 70) -> None:
    """Draw lines between matched keypoints in side-by-side images.

    Parameters
    ----------
    rgb1, rgb2 : np.ndarray
        RGB images, shapes (H1, W1, 3) and (H2, W2, 3).
    keypoints1, keypoints2 : np.ndarray
        Keypoints, shapes (N,3) and (M,3).
    matches : np.ndarray
        Matched index pairs, shape (K,2), dtype int.
    inliers : np.ndarray or None, optional
        Boolean RANSAC inlier mask, shape (K,); None draws all as tentative.
    max_lines : int, optional
        Maximum plotted matches, default 70.

    Returns
    -------
    None
        Displays a Matplotlib figure.
    """
    h1, w1 = rgb1.shape[:2]
    h2, w2 = rgb2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = rgb1
    canvas[:h2, w1:w1 + w2] = rgb2
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.imshow(canvas)
    if len(matches):
        selected = np.linspace(0, len(matches) - 1,
                               min(len(matches), max_lines), dtype=int)
        for i in selected:
            a, b = matches[i]
            x1, y1 = keypoints1[a, :2]
            x2, y2 = keypoints2[b, :2]
            good = inliers is not None and inliers[i]
            color = '#4ed7ad' if good else '#ffb15c'
            ax.plot([x1, w1 + x2], [y1, y2], color=color, alpha=0.68, linewidth=0.9)
    ax.set_title(f'{len(matches)} matches' +
                 (f' · {int(inliers.sum())} RANSAC inliers' if inliers is not None else ''))
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    

def pairwise_registration(rgb1: np.ndarray, rgb2: np.ndarray,
                      H: np.ndarray | None) -> np.ndarray | None:
    """Show the homography-warped image 1 beside image 2 and their overlay.

    Parameters
    ----------
    rgb1, rgb2 : np.ndarray
        Original images, shapes (H1,W1,3) and (H2,W2,3), dtype uint8.
    H : np.ndarray or None
        Homography shape (3,3), mapping image 1 into image 2,
        or None if registration failed.

    Returns
    -------
    np.ndarray
        The combined registered image.
        Displays a three-panel Matplotlib figure (or a failure message).
    """
    img_combined = None
    if H is None:
        print('Registration unavailable: no robust homography was estimated.')
        return img_combined
    h2, w2 = rgb2.shape[:2]
    warped = cv2.warpPerspective(rgb1, H, (w2, h2))
    valid = cv2.warpPerspective(np.full(rgb1.shape[:2], 255, dtype=np.uint8),
                                H, (w2, h2)) > 0
    overlay = rgb2.astype(np.float64)
    overlay[valid] = (0.5 * warped[valid].astype(np.float64) +
                      0.5 * rgb2[valid].astype(np.float64))
    img_combined=overlay.astype(np.uint8)

    return img_combined


def print_registration_summary(result: RegistrationResult, label: str = '') -> None:
    """Print keypoint, match and inlier counts for a registration.

    Parameters
    ----------
    result : RegistrationResult
        Pipeline output (keypoint arrays, descriptor arrays, matches, etc.).
    label : str, optional
        Label printed before statistics.

    Returns
    -------
    None
        Prints concise counts and success/failure status.
    """
    print(f'{label}: {len(result.keypoints1)} / {len(result.keypoints2)} keypoints, '
          f'{len(result.matches)} matches, {int(result.inliers.sum())} inliers; '
          f'homography: {"yes" if result.homography is not None else "no"}')


def plot_registration_process(
    rgb1: np.ndarray,
    rgb2: np.ndarray,
    result: RegistrationResult,
    label: str,
) -> None:
    """Report one sequential registration step."""
    del rgb1, rgb2
    print_registration_summary(result, label)
    
def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    """Encode linear-light values for standard sRGB display.

    Parameters
    ----------
    values : np.ndarray
        Linear-light values (usually shape (H, W) or (H, W, 3)).
        Values outside [0, 1] will be clipped.

    Returns
    -------
    encoded : np.ndarray
        sRGB-encoded values in [0, 1], same shape as ``values``,
        dtype float64.
    """
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    encoded = np.where(values <= 0.0031308,
                       12.92 * values,
                       1.055 * values ** (1.0 / 2.4) - 0.055)
    return encoded


def show_gray(ax: Axes, image: np.ndarray, title: str = '',
              linear: bool = True) -> None:
    """Display a grayscale image on a Matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Destination plotting axis.
    image : np.ndarray
        Grayscale image with shape (H, W), usually in [0, 1].
    title : str, optional
        Title of the panel.
    linear : bool, optional
        If True, encode linear-light values to sRGB before display;
        otherwise display values in [0, 1] directly.

    Returns
    -------
    None
        Draws the image on ``ax``; no array is returned.
    """
    view = linear_to_srgb(image) if linear else np.clip(image, 0, 1)
    ax.imshow(view, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis('off')


def image_histogram(ax: Axes, image: np.ndarray, title: str = '', *,
                    bins: int = 128) -> None:
    """Plot an image histogram in its numerical intensity domain.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Destination plotting axis.
    image : np.ndarray
        Image array, usually shape (H, W) with values in [0, 1].
        All elements are pooled into one histogram.
    title : str, optional
        Plot title.
    bins : int, optional
        Number of histogram bins, by default 128.

    Returns
    -------
    None
        Draws the histogram on ``ax``.
    """
    ax.hist(np.asarray(image).ravel(), bins=bins, range=(0, 1))
    ax.set(xlabel='Numerical value', ylabel='Pixel count', title=title)
    ax.set_xlim(0, 1)


def compare_gray(images: Sequence[np.ndarray], titles: Sequence[str], *,
                bins: int = 128) -> None:
    """Compare grayscale images and their histograms in a two-row figure.

    Parameters
    ----------
    images : sequence of np.ndarray
        One or more grayscale images, each shape (H, W), in [0, 1].
    titles : sequence of str
        One title per image; must have the same length as ``images``.
    bins : int, optional
        Number of histogram bins, by default 128.

    Returns
    -------
    None
        Displays a figure with shape (2 rows, len(images) columns)
        of Matplotlib axes; no image array is returned.
    """
    fig, axes = plt.subplots(2, len(images),
                             figsize=(4.2 * len(images), 6.5), squeeze=False)
    for i, (image, title) in enumerate(zip(images, titles)):
        show_gray(axes[0, i], image, title)
        image_histogram(axes[1, i], image, bins=bins)
    plt.tight_layout()
    plt.show()

# Build the registration tree from the bottom up.
# At each level, every pair becomes one node in the next level.
def under_tree(lst):
    registration_tree = [lst]
    level = lst
    level_number = 0

    while len(level) > 1:
        next_level = []

        i = 0
        while i + 1 < len(level):
            reference = reference_sift_pipeline(
                to_grayscale(level[i]),
                to_grayscale(level[i + 1]),
            )
            next_pair = pairwise_registration(
                level[i],
                level[i + 1],
                reference.homography,
            )

            if next_pair is None:
                raise RuntimeError(
                    f'Registration unavailable at level {level_number}: '
                    f'pair ({i}, {i + 1})'
                )

            next_level.append(next_pair)
            # To display all intermediate registered images, uncomment:
            # plt.imshow(next_pair)
            # plt.title(
            #     f'Registered overlay — level {level_number}, '
            #     f'node pair ({i}, {i + 1})'
            # )
            # plt.show()
            i += 2

        # An unmatched image is promoted unchanged to the next level.
        if i < len(level):
            next_level.append(level[i])

        registration_tree.append(next_level)
        level = next_level
        level_number += 1

    pairs = level

    # Display only the image left after all levels.
    final_image = pairs[0]
    plt.imshow(final_image)
    plt.title('Registered overlay — final image')
    plt.show()
    return final_image



def secuential(lst):
    # Sequentially register each image to the previous one.
    registered_images = [lst[0]]  # Start with the first image.

    for i in range(1, len(lst)):
        reference = reference_sift_pipeline(
            to_grayscale(registered_images[-1]),
            to_grayscale(lst[i]),
        )
        next_pair = pairwise_registration(
            registered_images[-1],
            lst[i],
            reference.homography,
        )

        if next_pair is None:
            raise RuntimeError(
                f'Registration unavailable at index {i}: '
                f'pair ({i - 1}, {i})'
            )

        registered_images.append(next_pair)
        plot_registration_process(
            registered_images[-2],
            lst[i],
            reference,
            f'Sequential — step {i}',
        )

    final_image = registered_images[-1]
    plt.imshow(final_image)
    plt.title('Registered overlay — final image')
    plt.show()

    return final_image


def plot_keypoints(rgb: np.ndarray, keypoints: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.imshow(rgb)
    for x, y, sigma in keypoints[:300]:
        ax.add_patch(Circle((x, y), max(2.0, 1.3 * sigma),
                            fill=False, linewidth=0.8, edgecolor='cyan',
                            alpha=0.72))
    ax.set_title(f'{title} ({len(keypoints)} keypoints)')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_registration(rgb1: np.ndarray, rgb2: np.ndarray,
                      homography: np.ndarray | None) -> None:
    if homography is None:
        print('Registration unavailable: no robust homography was estimated.')
        return
    h2, w2 = rgb2.shape[:2]
    warped = cv2.warpPerspective(rgb1, homography, (w2, h2))
    valid = cv2.warpPerspective(
        np.full(rgb1.shape[:2], 255, dtype=np.uint8),
        homography,
        (w2, h2),
    ) > 0
    overlay = rgb2.astype(np.float64)
    overlay[valid] = (0.5 * warped[valid] +
                       0.5 * rgb2[valid])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, image, title in zip(
        axes,
        (warped, rgb2, overlay.astype(np.uint8)),
        ('Warped image 1', 'Target image 2', 'Registered overlay'),
    ):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def main() -> None:
    image_paths = sorted(
        Path(__file__).parent.joinpath('cars_r').glob('cars-*.png'),
        key=lambda path: int(path.stem.split('-')[1]),
    )
    images = [read_rgb_image(path) for path in image_paths]
    # random.seed(42)  # for reproducibility
    # shuffle(images)   # shuffle images

    print(f'Loaded {len(images)} images.')
    first_reference = reference_sift_pipeline(
        to_grayscale(images[0]),
        to_grayscale(images[1]),
    )
    print_registration_summary(first_reference, 'OpenCV SIFT')
    plot_keypoints(images[0], first_reference.keypoints1,
                   'Reference keypoints — image 1')
    plot_matches(images[0], images[1], first_reference.keypoints1,
                 first_reference.keypoints2, first_reference.matches,
                 first_reference.inliers)
    plot_registration(images[0], images[1], first_reference.homography)

    print('Running tree registration...')
    under_tree(images)

    print('Running sequential registration...')
    secuential(images)


if __name__ == '__main__':
    main()