import numpy as np
import open3d as o3d
import sys

def get_color_map():
    # Complete SemanticKITTI color map (0-28)
    return {
        0: [0, 0, 0],          # "unlabeled"
        1: [245, 150, 100],    # "car"
        2: [245, 230, 100],    # "bicycle"
        3: [250, 80, 100],     # "motorcycle" 
        4: [150, 60, 30],      # "truck"
        5: [180, 30, 80],      # "other-vehicle"
        6: [255, 0, 0],        # "person"
        7: [30, 30, 255],      # "bicyclist"
        8: [200, 40, 255],     # "motorcyclist"
        9: [255, 0, 255],      # "road"
        10: [255, 150, 255],   # "parking"
        11: [75, 0, 75],       # "sidewalk"
        12: [75, 0, 175],      # "other-ground"
        13: [0, 200, 255],     # "building"
        14: [50, 120, 255],    # "fence"
        15: [0, 175, 0],       # "vegetation"
        16: [0, 60, 135],      # "trunk"
        17: [80, 240, 150],    # "terrain"
        18: [150, 240, 255],   # "pole"
        19: [0, 0, 255],       # "traffic-sign"
        20: [255, 255, 0],     # "other-object"
        21: [30, 30, 255],     # "moving-car"
        22: [255, 255, 50],    # "barrier"
        23: [250, 80, 100],    # "moving-motorcycle"
        24: [245, 150, 100],   # "moving-car"
        25: [150, 60, 30],     # "moving-truck"
        26: [255, 0, 0],       # "moving-person"
        27: [30, 30, 255],     # "moving-bicyclist"
        28: [200, 40, 255],    # "moving-motorcyclist"
    }

def get_learning_map():
    """Get learning map from raw labels to training labels"""
    return {
        0: 0,      # "unlabeled"
        1: 0,      # "outlier"
        10: 1,     # "car"
        11: 2,     # "bicycle"
        13: 5,     # "bus" -> "other-vehicle"
        15: 3,     # "motorcycle"
        16: 5,     # "on-rails" -> "other-vehicle"
        18: 4,     # "truck"
        20: 5,     # "other-vehicle"
        30: 6,     # "person"
        31: 7,     # "bicyclist"
        32: 8,     # "motorcyclist"
        40: 9,     # "road"
        44: 10,    # "parking"
        48: 11,    # "sidewalk"
        49: 12,    # "other-ground"
        50: 13,    # "building"
        51: 14,    # "fence"
        52: 0,     # "other-structure" -> "unlabeled"
        60: 9,     # "lane-marking" -> "road"
        70: 15,    # "vegetation"
        71: 16,    # "trunk"
        72: 17,    # "terrain"
        80: 18,    # "pole"
        81: 19,    # "traffic-sign"
        99: 20,    # "other-object"
        252: 1,    # "moving-car" -> "car"
        253: 7,    # "moving-bicyclist" -> "bicyclist"
        254: 6,    # "moving-person" -> "person"
        255: 8,    # "moving-motorcyclist" -> "motorcyclist"
        256: 5,    # "moving-on-rails" -> "other-vehicle"
        257: 5,    # "moving-bus" -> "other-vehicle"
        258: 4,    # "moving-truck" -> "truck"
        259: 5,    # "moving-other-vehicle" -> "other-vehicle"
    }

def colorize_point_cloud(points, labels):
    colormap = get_color_map()
    learning_map = get_learning_map()
    colors = np.zeros((len(labels), 3))
    
    # SemanticKITTI labels are stored in uint32
    semantic_labels = labels & 0xFFFF  # Extract lower 16 bits
    
    # First map raw labels to training labels
    remapped_labels = np.zeros_like(semantic_labels)
    for raw_label, train_label in learning_map.items():
        mask = semantic_labels == raw_label
        remapped_labels[mask] = train_label
    
    # Then apply colors
    for label_id, color in colormap.items():
        mask = remapped_labels == label_id
        colors[mask] = np.array(color) / 255.0
    
    return colors

def visualize_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Add coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    # Set initial viewing angle
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(coordinate_frame)
    
    # Set background to white (optional)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python vis.py <path_to_point_cloud.bin> <path_to_labels.label>")
        sys.exit(1)

    point_cloud_path = sys.argv[1]
    label_path = sys.argv[2]

    # Load point cloud
    points = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, 4)
    
    # Load labels
    labels = np.fromfile(label_path, dtype=np.uint32)
    
    # Verify data shapes
    assert len(points) == len(labels), "Points and labels count mismatch"
    
    # Colorize the point cloud
    colors = colorize_point_cloud(points, labels)
    
    # Visualize the result
    visualize_point_cloud(points, colors)
