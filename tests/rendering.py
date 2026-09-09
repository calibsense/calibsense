"""Rendering synthetic calibration images.

Test infrastructure, not part of the package. Detectors can only be tested
honestly against pixels, so this projects a board texture through a known camera
— distortion included — and hands back an image OpenCV has to find the pattern
in on its own.

The distortion is applied by resampling: for every pixel of the *distorted*
output, the matching position in the undistorted image is found analytically,
which is the direction `undistortPoints` inverts and therefore exact rather than
iterated.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from caltrust.core.camera import CameraModel, FisheyeKannalaBrandt
from caltrust.core.poses import Pose
from caltrust.core.target import CharucoBoard, Checkerboard, CircleGrid, TargetSpec

#: Texture resolution, in pixels per millimetre of board.
TEXTURE_SCALE = 4.0

#: Quiet margin drawn around the pattern, in millimetres.
BOARD_MARGIN_MM = 12.0


def board_texture(target: TargetSpec) -> Tuple[np.ndarray, np.ndarray]:
    """Render a board face-on, and say where its corners are in the texture.

    Args:
        target: The target to draw.

    Returns:
        The texture image and a `(4, 2)` array of texture pixel coordinates for
        the board-frame rectangle `[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]`
        that bounds the pattern plus its margin.
    """
    points = target.object_points()
    low = points[:, :2].min(axis=0) - BOARD_MARGIN_MM
    high = points[:, :2].max(axis=0) + BOARD_MARGIN_MM
    width = int(round((high[0] - low[0]) * TEXTURE_SCALE))
    height = int(round((high[1] - low[1]) * TEXTURE_SCALE))
    image = np.full((height, width), 255, np.uint8)

    def to_texture(xy_mm):
        return (np.asarray(xy_mm, dtype=float) - low) * TEXTURE_SCALE

    if isinstance(target, Checkerboard):
        step = points[1, 0] - points[0, 0]
        # Inner corners sit at multiples of `step`; squares are offset by one
        # half-period so that the corner grid lands where the detector expects.
        for row in range(-1, target.rows + 1):
            for column in range(-1, target.columns + 1):
                if (row + column) % 2:
                    continue
                corner = to_texture([(column - 0.5) * step, (row - 0.5) * step])
                far = to_texture([(column + 0.5) * step, (row + 0.5) * step])
                cv2.rectangle(
                    image,
                    tuple(np.round(corner).astype(int)),
                    tuple(np.round(far).astype(int)),
                    0, -1,
                )
    elif isinstance(target, CircleGrid):
        spacing = float(np.linalg.norm(points[1, :2] - points[0, :2]))
        radius = max(int(round(0.28 * spacing * TEXTURE_SCALE)), 3)
        for point in points:
            centre = np.round(to_texture(point[:2])).astype(int)
            cv2.circle(image, tuple(centre), radius, 0, -1, lineType=cv2.LINE_AA)
    elif isinstance(target, CharucoBoard):
        board = cv2.aruco.CharucoBoard(
            (target.squares_x, target.squares_y),
            target.square_size, target.marker_size,
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, target.dictionary)),
        )
        board.setLegacyPattern(target.legacy_pattern)
        span = np.array([target.squares_x, target.squares_y]) * target.square_size
        drawn = board.generateImage(
            tuple(np.round(span * TEXTURE_SCALE).astype(int)), marginSize=0
        )
        origin = np.round(to_texture([0.0, 0.0])).astype(int)
        image[
            origin[1] : origin[1] + drawn.shape[0],
            origin[0] : origin[0] + drawn.shape[1],
        ] = drawn
    else:
        raise TypeError(f"no texture renderer for {type(target).__name__}")

    rectangle = np.array([
        [low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]
    ])
    return image, rectangle


def _undistorted_positions(
    camera: CameraModel, size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    width, height = size
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)
    if isinstance(camera, FisheyeKannalaBrandt):
        deskewed = pixels.copy()
        deskewed[:, 0] -= (
            camera.alpha * camera.fx * (pixels[:, 1] - camera.cy) / camera.fy
        )
        skewless = camera.camera_matrix.copy()
        skewless[0, 1] = 0.0
        normalised = cv2.fisheye.undistortPoints(
            np.ascontiguousarray(deskewed.reshape(-1, 1, 2)), skewless, camera.distortion
        ).reshape(-1, 2)
    else:
        normalised = cv2.undistortPoints(
            np.ascontiguousarray(pixels.reshape(-1, 1, 2)),
            camera.camera_matrix, camera.distortion,
        ).reshape(-1, 2)
    ideal = np.stack([
        normalised[:, 0] * camera.fx + camera.cx,
        normalised[:, 1] * camera.fy + camera.cy,
    ], axis=1)
    return (
        ideal[:, 0].reshape(height, width).astype(np.float32),
        ideal[:, 1].reshape(height, width).astype(np.float32),
    )


def render_view(
    camera: CameraModel,
    target: TargetSpec,
    pose: Pose,
    image_size: Tuple[int, int] = (1280, 720),
    background: int = 190,
    blur: float = 0.0,
    noise: float = 0.0,
    seed: Optional[int] = 0,
) -> np.ndarray:
    """Render one view of a target through a known camera.

    Args:
        camera: The intrinsics to render through, distortion included.
        target: The target to render.
        pose: Board-to-camera transform.
        image_size: Output size as `(width, height)`.
        background: Grey level outside the board.
        blur: Gaussian blur sigma in pixels, for simulating defocus.
        noise: Gaussian pixel noise standard deviation, in grey levels.
        seed: Seed for the noise.

    Returns:
        An 8-bit grayscale image.
    """
    texture, rectangle = board_texture(target)
    points = target.object_points()
    low = points[:, :2].min(axis=0) - BOARD_MARGIN_MM
    corners_texture = (rectangle - low) * TEXTURE_SCALE

    # Project the board rectangle with a pinhole model at the same focal length,
    # then apply distortion by resampling; a homography cannot carry distortion.
    in_camera = pose.apply(np.column_stack([rectangle, np.zeros(4)]))
    ideal = np.stack([
        in_camera[:, 0] / in_camera[:, 2] * camera.fx + camera.cx,
        in_camera[:, 1] / in_camera[:, 2] * camera.fy + camera.cy,
    ], axis=1)

    homography = cv2.getPerspectiveTransform(
        corners_texture.astype(np.float32), ideal.astype(np.float32)
    )
    width, height = image_size
    undistorted = cv2.warpPerspective(
        texture, homography, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=background,
    )
    map_x, map_y = _undistorted_positions(camera, image_size)
    image = cv2.remap(
        undistorted, map_x, map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=background,
    )
    if blur > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur)
    if noise > 0:
        rng = np.random.default_rng(seed)
        image = np.clip(
            image.astype(np.float64) + rng.normal(0.0, noise, image.shape), 0, 255
        ).astype(np.uint8)
    return image


def write_views(
    directory: str,
    camera: CameraModel,
    target: TargetSpec,
    poses,
    image_size: Tuple[int, int] = (1280, 720),
    prefix: str = "view",
    **kwargs,
) -> Tuple[str, ...]:
    """Render several views and write them as PNG files.

    Args:
        directory: Destination directory, created if missing.
        camera: The intrinsics to render through.
        target: The target to render.
        poses: Board-to-camera poses.
        image_size: Output size as `(width, height)`.
        prefix: Filename prefix.
        **kwargs: Passed to `render_view`.

    Returns:
        The paths written, in order.
    """
    import os

    os.makedirs(directory, exist_ok=True)
    paths = []
    for index, pose in enumerate(poses):
        image = render_view(camera, target, pose, image_size, **kwargs)
        path = os.path.join(directory, f"{prefix}{index:03d}.png")
        cv2.imwrite(path, image)
        paths.append(path)
    return tuple(paths)
