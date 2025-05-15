import cv2
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform

class Fiducial:
    def __init__(self, video):
        """
        Initialize the Fiducial class with a sequence of frames.
        Automatically converts RGB video to grayscale if needed.

        Args:
            video (numpy.ndarray): A 3D or 4D array representing the video frames.
                - Shape (num_frames, height, width) for grayscale.
                - Shape (num_frames, height, width, 3) for RGB.

        Raises:
            TypeError: If input is not a NumPy array.
            ValueError: If input is empty or has unsupported dimensions.
        """
        
        if not isinstance(video, np.ndarray):
            raise TypeError("Input 'video' must be a NumPy array.")

        if video.size == 0:
            raise ValueError("Input 'video' is empty.")

        if video.ndim == 4 and video.shape[-1] == 3:
            # RGB video: convert each frame to grayscale
            self.video = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in video])
        elif video.ndim == 3:
            # Already grayscale
            self.video = video
        else:
            raise ValueError(
                f"Invalid video shape: {video.shape}. Expected 3D (grayscale) or 4D (RGB) array."
            )
        
    def pre_process(self, clip_range=None, enhance_contrast=False,
                    clahe=False, apply_blur=False,
                    adaptive_threshold=False, morph_operations=False,
                    blur_kernel_size=(5, 5)):
        """
        Apply preprocessing to all frames in the video to improve marker detection.

        Args:
            clip_range (tuple or None): (min, max) for clipping and normalization.
            enhance_contrast (bool): Apply global histogram equalization.
            clahe (bool): Apply adaptive histogram equalization (CLAHE).
            apply_blur (bool): Apply Gaussian blur.
            adaptive_threshold (bool): Apply adaptive thresholding.
            morph_operations (bool): Apply morphological operations (open and close).
            blur_kernel_size (tuple): Kernel size for Gaussian blur (default (5, 5)).

        Returns:
            numpy.ndarray: Preprocessed grayscale images.
        """
        processed_video = []

        if clip_range is not None:
            if not (isinstance(clip_range, (list, tuple)) and len(clip_range) == 2):
                raise ValueError("clip_range must be a tuple/list of (min, max)")

        for frame in tqdm(self.video, dynamic_ncols=True, desc="Preprocessing input video"):
            gray = frame.copy()

            if clip_range is not None:
                clip_min, clip_max = clip_range
                gray = np.clip(gray, clip_min, clip_max)
                gray = ((gray - clip_min) / (clip_max - clip_min) * 255).astype("uint8")

            if enhance_contrast:
                gray = cv2.equalizeHist(gray)

            if clahe:
                clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe_obj.apply(gray)

            if apply_blur:
                gray = cv2.GaussianBlur(gray, blur_kernel_size, 0)

            if adaptive_threshold:
                gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

            if morph_operations:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
                gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

            processed_video.append(gray)

        return processed_video
    
    def determine_aruco(self, frame):
        """
        Determine the ArUco dictionary used in a given frame.

        Args:
            frame (numpy.ndarray): Input frame (grayscale)

        Returns:
            str: Name of the detected ArUco dictionary.

        Raises:
            ValueError: If no valid dictionary could detect markers in the frame.
        """
        ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
        }

        # ArUco marker detector parameters
        parameters = cv2.aruco.DetectorParameters()

        # Try detecting markers with each dictionary
        for name, aruco_id in ARUCO_DICT.items():
            dictionary = cv2.aruco.getPredefinedDictionary(aruco_id)
            corners, _, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
            
            if corners:
                return name

        # Raise error if no dictionary could detect any marker
        raise ValueError("No valid ArUco dictionary matched the input frame.")

    def detect_markers(self, video=None, marker_type="aruco", fiducial_dictionary=None, known_ids=None):
        """
        Detect fiducial markers in the preprocessed video or original input video.

        Args:
            video (list of numpy.ndarray or None): Optional video to detect markers in. If None, uses self.video.
            marker_type (str): One of ["aruco", "apriltag", "charuco", "artoolkit"].
            fiducial_dictionary (str or None): Dictionary name (e.g., for ArUco or AprilTag).
            known_ids (list or None): If provided, only markers with these IDs are considered.

        Returns:
            list: Detected marker information per frame, or "none" if none detected.

        Raises:
            ValueError: If inputs are invalid or unsupported marker type is used.
        """
        if video is None:
            video = self.video

        if not isinstance(video, list) or len(video) == 0:
            raise ValueError("Video must be a non-empty list of frames (NumPy arrays).")
        
        results = []
        frame_success = 0
        total_markers = 0

        if marker_type == "aruco":
            if fiducial_dictionary is None:
                fiducial_dictionary = self.determine_aruco(video[0])
                if fiducial_dictionary is None:
                    raise ValueError("No ArUco dictionary could be determined from the first frame.")

            dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, fiducial_dictionary))
            parameters = cv2.aruco.DetectorParameters()

            for i, frame in tqdm(enumerate(video), total=len(video), dynamic_ncols=True, desc="Detection Progress of Fiducial Markers"):
                corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

                if ids is not None:
                    marker_info = []
                    for marker_id, corner in zip(ids.flatten(), corners):
                        if known_ids and marker_id not in known_ids:
                            continue
                        detected_corners = [(int(c[1]), int(c[0])) for c in corner[0]]
                        marker_info.append((int(marker_id), detected_corners))

                    marker_info.sort(key=lambda x: x[0])

                    if marker_info:
                        results.append(marker_info)
                        frame_success += 1
                        total_markers += len(marker_info)
                    else:
                        results.append("none")
                else:
                    results.append("none")

        elif marker_type == "apriltag":

            raise ValueError(f"Marker family {marker_type} will be supported soon.")
        
        elif marker_type == "charuco":
           
            raise ValueError(f"Marker family {marker_type} will be supported soon.")

        elif marker_type == "artoolkit":

            raise ValueError(f"Marker family {marker_type} will be supported soon.")

        else:
            raise ValueError(f"Unsupported marker family: {marker_type}")

        
        success_rate = (frame_success / max(1, len(video))) * 100
        print(f"Detection Success Rate: {round(success_rate, 2)}%")

        if success_rate < 100:
            warnings.warn(
                "Detection Success Rate is below 100%. "
                "This means that some frames are missing markers, "
                "which may result in unreliable tracking. "
                "Consider re-running preprocessing or improving image quality.",
                category=UserWarning
            )

        return results
    
    @staticmethod
    def get_common_markers(reference_frame_markers, target_frame_markers):
        """
        Extracts and matches common fiducial markers between two frames.

        Args:
            reference_frame_markers (list): Marker list from the reference frame, formatted as [(id, corners), ...].
            target_frame_markers (list): Marker list from the current frame, formatted as [(id, corners), ...].

        Returns:
            tuple: (src_points, dst_points) as np.ndarray of shape (N, 2), or (None, None) if no common markers.

        Raises:
            ValueError: If either input is None or not a list of tuples.
        """
        # Input validation
        if not isinstance(reference_frame_markers, list) or not isinstance(target_frame_markers, list):
            raise ValueError("Both reference_frame_markers and target_frame_markers must be lists.")
        if not all(isinstance(m, tuple) and isinstance(m[1], (list, np.ndarray)) for m in reference_frame_markers + target_frame_markers):
            raise ValueError("Each marker entry must be a tuple (id, corners), with corners as list or ndarray.")

        # Convert marker lists to dictionaries keyed by ID
        f0 = {id_: np.array(corners, dtype=np.float32) for id_, corners in reference_frame_markers}
        f1 = {id_: np.array(corners, dtype=np.float32) for id_, corners in target_frame_markers}
        
        # Find shared marker IDs
        common_ids = sorted(set(f0) & set(f1))

        # If no common markers are found, return None
        if not common_ids:
            return None, None

        # Collect matched corner points
        src = np.vstack([f0[i] for i in common_ids])  # Points from reference frame
        dst = np.vstack([f1[i] for i in common_ids])  # Corresponding points from target frame

        return src, dst

    @staticmethod
    def compute_transformation_matrix(source_points, destination_points, transform_type="euclidean"):
        """
        Computes a transformation matrix between source and destination points using the specified model.

        Args:
            source_points (np.ndarray): Source points from reference frame. Shape (N, 2).
            destination_points (np.ndarray): Destination points from target frame. Shape (N, 2).
            transform_type (str): One of "euclidean", "affine", or "homography".

        Returns:
            np.ndarray: 3x3 transformation matrix.

        Raises:
            ValueError: If inputs are invalid or if transformation cannot be computed.
        """
        # Validate input type and shape
        if source_points is None or destination_points is None:
            raise ValueError("Source and destination points must not be None.")
        
        if not isinstance(source_points, np.ndarray) or not isinstance(destination_points, np.ndarray):
            raise ValueError("Source and destination points must be NumPy arrays.")
        
        if source_points.shape != destination_points.shape or source_points.shape[1] != 2:
            raise ValueError("Source and destination points must have the same shape (N, 2).")

        # Estimate transformation based on the specified type
        if transform_type == "euclidean":
            M, _ = cv2.estimateAffinePartial2D(destination_points, source_points, method=cv2.RANSAC)
        elif transform_type == "affine":
            M, _ = cv2.estimateAffine2D(destination_points, source_points, method=cv2.RANSAC)
        elif transform_type == "homography":
            M, _ = cv2.findHomography(destination_points, source_points, method=cv2.RANSAC)
        else:
            raise ValueError("Invalid transform_type. Choose from 'euclidean', 'affine', or 'homography'.")

        # Handle failure to estimate matrix
        if M is None:
            raise ValueError("Transformation matrix could not be computed. Check your input points.")

        # Convert 2x3 affine matrix to 3x3 homogeneous matrix
        if M.shape[0] == 2 and M.shape[1] == 3:
            M = np.vstack([M, [0, 0, 1]])

        return M

    def compute_transformations(self, id_coords, reference_index=0, transform_type="euclidean"):
        """
        Computes transformation matrices from each frame to the reference frame based on common fiducial markers.

        Args:
            id_coords (list): List of (marker_id, corner_coordinates) tuples or 'none'.
            reference_index (int): Index of the reference frame to align others to.
            transform_type (str): Type of transformation: "euclidean", "affine", or "homography".

        Returns:
            list: List of 3x3 transformation matrices. If transformation fails for a frame, its entry will be None.
        """
        self.reference_index = reference_index

        if not isinstance(id_coords, list) or len(id_coords) == 0:
            raise ValueError("Input 'id_coords' must be a non-empty list of marker detections.")

        if not (0 <= self.reference_index < len(id_coords)):
            raise ValueError("Invalid reference_index. It must be within the range of id_coords.")

        if id_coords[self.reference_index] == "none":
            raise ValueError(f"The reference frame at index {self.reference_index} has no marker data.")

        n = len(id_coords)
        transformations = [None] * n
        transformations[self.reference_index ] = np.eye(3)  # Identity matrix for reference

        for i in range(n):
            if i == self.reference_index :
                continue

            if id_coords[i] == "none":
                warnings.warn(f"[Frame {i}] No fiducial markers detected. Transformation will be set to None.")
                continue

            source_points, destination_points = Fiducial.get_common_markers(
                id_coords[self.reference_index ], id_coords[i]
            )

            if (
                source_points is None or destination_points is None or
                len(source_points) < 1 or len(destination_points) < 1
            ):
                warnings.warn(f"[Frame {i}] No common markers with reference frame {self.reference_index}. Transformation will be set to None.")
                continue

            source_points = np.array(source_points)
            destination_points = np.array(destination_points)

            if source_points.shape != destination_points.shape:
                raise ValueError(f"[Frame {i}] Source and destination points have different shapes: "
                                f"{source_points.shape} vs {destination_points.shape}.")

            M = Fiducial.compute_transformation_matrix(source_points, destination_points, transform_type)
            transformations[i] = M

        return transformations

    @staticmethod
    def revert_fiducial(id_coords, transformations):
        """
        Applies inverse transformations to marker coordinates in each frame to compensate for global motion,
        aligning all markers to the reference frame's coordinate system.

        Args:
            id_coords (list): List of tuples (marker_id, corner_coordinates).
            transformations (list): List of 3x3 transformation matrices aligning each frame to the reference frame.
                                    If a matrix is None, the corresponding frame is returned unchanged.

        Returns:
            list: List of transformed fiducial markers id and coordinates
        """
        if not isinstance(id_coords, list) or not isinstance(transformations, list):
            raise ValueError("Both 'id_coords' and 'transformations' must be lists.")

        if len(id_coords) != len(transformations):
            raise ValueError("Length mismatch: 'id_coords' and 'transformations' must be the same length.")

        transformed_fiducial = []

        for idx, (id_coords, M) in enumerate(zip(id_coords, transformations)):
            if M is None:
                # No transformation available; keep original
                print(f"[Frame {idx}] No transformation found, returning original.")
                transformed_fiducial.append(id_coords)
                continue

            try:
                compensated_frame = []
                for marker_id, corners in id_coords:
                    corners_arr = np.array(corners, dtype=np.float32)

                    if corners_arr.shape[1] != 2:
                        raise ValueError(f"Invalid corner shape for marker {marker_id}: {corners_arr.shape}")

                    # Convert to homogeneous coordinates: (x, y) -> (x, y, 1)
                    corners_hom = np.column_stack([corners_arr, np.ones(len(corners_arr), dtype=np.float32)])

                    # Apply transformation: M * point
                    transformed = (M @ corners_hom.T).T  # shape: (N, 3)

                    # Normalize if using homography
                    if transformed.shape[1] == 3 and not np.allclose(transformed[:, 2], 1):
                        transformed = transformed[:, :2] / transformed[:, 2].reshape(-1, 1)
                    else:
                        transformed = transformed[:, :2]

                    compensated_frame.append((marker_id, transformed.tolist()))

                transformed_fiducial.append(compensated_frame)

            except Exception as e:
                print(f"[Frame {idx}] Error while transforming markers: {e}")
                transformed_fiducial.append(id_coords)  # Fallback to original

        return transformed_fiducial

    def revert_frames(self, transformations, transform_type="euclidean", use_interpolation=True):
        """
        Applies inverse transformations to warp all frames into the coordinate system of frame 0.

        Args:
            transformations (list): List of 3x3 transformation matrices mapping each frame to frame 0.
            transform_type (str): Type of transformation ("euclidean", "affine", or "homography").
            use_interpolation (bool): True for bilinear interpolation, False for nearest neighbor.

        Returns:
            np.ndarray: Aligned 3D array of the same shape as data_array.
        """
        if not isinstance(self.video, np.ndarray) or self.video.ndim != 3:
            raise ValueError("'data_array' must be a 3D NumPy array (frames, rows, cols).")

        if transform_type not in {"euclidean", "affine", "homography"}:
            raise ValueError("Invalid 'transform_type'. Must be 'euclidean', 'affine', or 'homography'.")

        num_frames, rows, cols = self.video.shape

        if len(transformations) != num_frames:
            raise ValueError("Length of 'transformations' must match number of frames in 'data_array'.")

        aligned_frames = np.full_like(self.video, np.nan)

        for i in tqdm(range(num_frames),  dynamic_ncols=True, desc="Reverting frames to reference"):
            M = transformations[i]

            if M is None:
                print(f"[Frame {i}] No transformation available. Skipping.")
                continue

            try:
                # Invert transformation
                M_inv = np.linalg.inv(M)

                if transform_type in {"euclidean", "affine"}:
                    # Extract affine parameters
                    R = M_inv[:2, :2]
                    t = M_inv[:2, 2]

                    aligned_frames[i] = affine_transform(
                        self.video[i], R, offset=t,
                        order=1 if use_interpolation else 0,
                        mode='nearest'
                    )

                elif transform_type == "homography":
                    aligned_frames[i] = cv2.warpPerspective(
                        self.video[i], M_inv, (cols, rows),
                        flags=cv2.INTER_LINEAR if use_interpolation else cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=np.nan
                    )

            except np.linalg.LinAlgError:
                print(f"[Frame {i}] Failed to invert transformation matrix. Skipping.")
                continue
            except Exception as e:
                print(f"[Frame {i}] Error during warping: {e}")
                continue

        return aligned_frames

    def uncertainty_analysis(self, id_coords, transformed_fiducial, plot=False, time_axis=None, save_path=None):
        """
        Computes per-frame and overall Euclidean error statistics between transformed and reference fiducial markers.

        Args:
            id_coords (list): List of tuples [(id, corners)] for each frame, representing original marker positions.
            transformed_fiducial (list): List of tuples [(id, corners)] for each frame, representing transformed marker positions.
            plot (bool): Whether to plot the results over time/frame index.
            time_axis (list or None): Optional x-axis values (e.g. time in seconds) for plotting.
            save_path (str or None): Optional path to save the plot.

        Returns:
            dict: A dictionary with:
                - "Mean Error": Mean Euclidean distance error across all frames (excluding NaNs).
                - "Standard Deviation": Standard deviation of per-frame mean errors (excluding NaNs).

        Raises:
            ValueError: If input lists are mismatched in length or the reference frame is invalid.
            AttributeError: If `reference_index` is not defined in the class.
        """
        
        def parse_frame_data(frame_data, frame_idx):
            """Convert list of (id, corners) to {id: np.array(corners)}."""
            if frame_data == "none" or not frame_data:
                return None
            try:
                return {mid: np.array(corners, dtype=np.float32) for mid, corners in frame_data}
            except Exception as e:
                warnings.warn(f"[Frame {frame_idx}] Failed to parse frame data: {e}")
                return None

        n_frames = len(id_coords)
        if n_frames != len(transformed_fiducial):
            raise ValueError("Length mismatch between original and transformed fiducial lists.")

        if not hasattr(self, "reference_index"):
            raise AttributeError("reference_index not set. Run compute_transformations() first.")

        ref_idx = self.reference_index
        reference_data = parse_frame_data(id_coords[ref_idx], ref_idx)
        if reference_data is None:
            raise ValueError("Reference frame is invalid or contains no markers.")

        frame_means = []
        frame_stds = []

        for i in range(n_frames):
            if i == ref_idx:
                frame_means.append(0.0)
                frame_stds.append(0.0)
                continue

            orig = parse_frame_data(id_coords[i], i)
            trans = parse_frame_data(transformed_fiducial[i], i)

            if orig is None or trans is None:
                frame_means.append(np.nan)
                frame_stds.append(np.nan)
                continue

            common_ids = set(reference_data.keys()) & set(trans.keys())
            if not common_ids:
                warnings.warn(f"[Frame {i}] No common markers with reference frame.")
                frame_means.append(np.nan)
                frame_stds.append(np.nan)
                continue

            errors = []
            for mid in common_ids:
                ref_corners = reference_data[mid]
                trans_corners = trans[mid]

                if ref_corners.shape != trans_corners.shape:
                    warnings.warn(f"[Frame {i}] Shape mismatch for marker {mid}.")
                    continue

                dists = np.linalg.norm(ref_corners - trans_corners, axis=1)  # per-corner error
                errors.extend(dists)

            if errors:
                frame_means.append(np.mean(errors))
                frame_stds.append(np.std(errors))
            else:
                frame_means.append(np.nan)
                frame_stds.append(np.nan)

        # X-axis values for plot
        x_vals = time_axis if time_axis is not None else list(range(n_frames))

        # Global statistics
        overall_mean = np.nanmean(frame_means)
        overall_std = np.nanstd(frame_means)

        # Optional plot
        if plot:
            try:
                self.plot_uncertainty_results(
                    x_values=x_vals,
                    frame_means=frame_means,
                    frame_stds=frame_stds,
                    overall_mean=overall_mean,
                    overall_std=overall_std,
                    time_axis=time_axis,
                    save_path=save_path
                )
            except Exception as e:
                print(f"[Plot Error] Failed to generate plot: {e}")

        return {
            "Mean Error": overall_mean,
            "Standard Deviation": overall_std
        }


    def plot_uncertainty_results(self, x_values, frame_means, frame_stds,
                                 overall_mean,overall_std, time_axis=None, save_path=None):
        """
        Plots uncertainty results showing per-frame mean error with standard deviation shading
        and overall statistics.

        Args:
        - x_values: List of frame indices or time values.
        - frame_means: List of per-frame mean errors.
        - frame_stds: List of per-frame standard deviations.
        - overall_mean: Overall mean error.
        - overall_std: Overall standard deviation.
        - time_axis: Optional list of time values to replace x_values (e.g., frame_idx / fps).
        - save_path: If provided, saves the figure to this path.
        """
        x = time_axis if time_axis is not None else x_values

        plt.figure(figsize=(8, 5))

        # Per-frame mean error and standard deviation shading
        plt.plot(x, frame_means, marker='o', linestyle='-', color='#ffb6c1', linewidth=0.5, label="Per-frame error")
        plt.fill_between(x,
                         np.array(frame_means) - np.array(frame_stds),
                         np.array(frame_means) + np.array(frame_stds),
                         color='#ffb6c1', alpha=0.5, label=r"Per-frame std. dev.")

        # Overall mean and std bands
        plt.axhline(y=overall_mean, color='#1f77b4', linestyle='dashed', linewidth=2.5, label="Overall mean error")
        plt.axhline(y=overall_mean + overall_std, color='#1f77b4', linestyle='dotted', linewidth=2,
                    label=r"Overall std. dev.")
        plt.axhline(y=overall_mean - overall_std, color='#1f77b4', linestyle='dotted', linewidth=2)

        # Formatting
        plt.xlabel("Time [s]" if time_axis is not None else "Frame index", fontsize=14)
        plt.ylabel("Transformation Error [px]", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(bottom=0)
        plt.legend(fontsize=10, ncol=2, loc='upper left')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=600)
        plt.show()