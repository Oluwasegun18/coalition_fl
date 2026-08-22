import numpy as np

def balanced_dirichlet_split(x, y, num_clusters=3, devices_per_cluster=10, alpha=0.5, seed=42):
    np.random.seed(seed)
    total_devices = num_clusters * devices_per_cluster
    labels = np.unique(y)
    
    # Create list of indices for each class
    class_indices = {label: np.where(y == label)[0] for label in labels}
    for indices in class_indices.values():
        np.random.shuffle(indices)

    # Allocate samples using Dirichlet distribution for each class
    device_indices = defaultdict(list)
    for label in labels:
        indices = class_indices[label]
        proportions = np.random.dirichlet([alpha] * total_devices)
        # Round proportionally
        proportions = (proportions * len(indices)).astype(int)
        proportions[-1] += len(indices) - proportions.sum()  # fix rounding

        start = 0
        for device_id, count in enumerate(proportions):
            end = start + count
            device_indices[device_id].extend(indices[start:end])
            start = end

    # Shuffle within each device and organize by cluster
    cluster_device_data = {}
    cluster_device_labels = {}
    for c in range(num_clusters):
        cluster_device_data[c] = {}
        cluster_device_labels[c] = {}
        for d in range(devices_per_cluster):
            device_id = c * devices_per_cluster + d
            dev_idx = device_indices[device_id]
            np.random.shuffle(dev_idx)
            cluster_device_data[c][d] = x[dev_idx]
            cluster_device_labels[c][d] = y[dev_idx]

    return cluster_device_data, cluster_device_labels


from collections import defaultdict
import numpy as np

def cluster_partition_disjoint_labels(x, y, num_clusters=3, devices_per_cluster=10, seed=42):
    np.random.seed(seed)
    total_devices = num_clusters * devices_per_cluster
    assert len(np.unique(y)) == devices_per_cluster, "Each cluster must have number of labels equal to devices per cluster"

    labels = np.unique(y)
    label_to_indices = {label: np.where(y == label)[0] for label in labels}

    # Shuffle each class's indices and split into cluster parts
    cluster_label_indices = {c: {} for c in range(num_clusters)}
    for label in labels:
        indices = label_to_indices[label]
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_clusters)
        for c in range(num_clusters):
            cluster_label_indices[c][label] = splits[c]

    # Assign labels to devices in each cluster (one label per device)
    cluster_device_data = {c: {} for c in range(num_clusters)}
    cluster_device_labels = {c: {} for c in range(num_clusters)}

    for c in range(num_clusters):
        class_list = list(cluster_label_indices[c].keys())
        np.random.shuffle(class_list)  # shuffle class assignments to devices

        for d in range(devices_per_cluster):
            label = class_list[d]
            data_indices = cluster_label_indices[c][label]
            cluster_device_data[c][d] = x[data_indices]
            cluster_device_labels[c][d] = y[data_indices]

    return cluster_device_data, cluster_device_labels



import numpy as np
from collections import defaultdict

def complementary_partition(x, y, num_clusters=3, devices_per_cluster=10, classes_per_device=2, seed=42):
    np.random.seed(seed)
    total_devices = num_clusters * devices_per_cluster
    labels = np.unique(y)
    label_to_indices = {label: np.where(y == label)[0] for label in labels}
    
    for indices in label_to_indices.values():
        np.random.shuffle(indices)

    # Preallocate
    cluster_device_data = {c: {} for c in range(num_clusters)}
    cluster_device_labels = {c: {} for c in range(num_clusters)}
    
    # Track how many samples to give per device
    samples_per_device = len(y) // total_devices
    samples_per_class = {label: len(label_to_indices[label]) for label in labels}

    # Rotate classes across clusters for variety
    class_lists = []
    for c in range(num_clusters):
        rotated = np.roll(labels, c * classes_per_device)
        class_lists.append(rotated.reshape(-1, classes_per_device))

    idx_ptrs = {label: 0 for label in labels}
    for c in range(num_clusters):
        for d in range(devices_per_cluster):
            device_id = c * devices_per_cluster + d
            device_classes = class_lists[c][d % len(class_lists[c])]
            
            indices = []
            quota = samples_per_device // classes_per_device
            for cls in device_classes:
                start = idx_ptrs[cls]
                end = start + quota
                indices.extend(label_to_indices[cls][start:end])
                idx_ptrs[cls] += quota
            
            np.random.shuffle(indices)
            cluster_device_data[c][d] = x[indices]
            cluster_device_labels[c][d] = y[indices]
    
    return cluster_device_data, cluster_device_labels



import numpy as np
from collections import defaultdict

def partition_mnist_clusterwise_complementary(x, y, num_clusters=3, devices_per_cluster=10, classes_per_device=2, seed=42):
    assert classes_per_device * devices_per_cluster >= 10, \
        "Not enough label slots to cover all 10 labels per cluster"
    
    np.random.seed(seed)
    total_devices = num_clusters * devices_per_cluster
    labels = np.unique(y)
    label_to_indices = {label: np.where(y == label)[0] for label in labels}
    for v in label_to_indices.values():
        np.random.shuffle(v)

    idx_ptr = {label: 0 for label in labels}
    samples_per_device = len(y) // total_devices
    samples_per_class = samples_per_device // classes_per_device

    cluster_device_data = {}
    cluster_device_labels = {}

    for c in range(num_clusters):
        cluster_device_data[c] = {}
        cluster_device_labels[c] = {}

        # Track label assignment count to ensure full label coverage
        label_pool = list(labels)
        np.random.shuffle(label_pool)

        # Step 1: Assign labels to devices ensuring full label coverage
        label_assignments = [[] for _ in range(devices_per_cluster)]

        # First assign each label at least once across the devices
        for i, label in enumerate(label_pool):
            label_assignments[i % devices_per_cluster].append(label)

        # Step 2: Fill remaining slots randomly from label pool
        for i in range(devices_per_cluster):
            while len(label_assignments[i]) < classes_per_device:
                choice = np.random.choice(label_pool)
                if choice not in label_assignments[i]:
                    label_assignments[i].append(choice)

        # Step 3: Assign samples based on labels
        for d in range(devices_per_cluster):
            device_labels = label_assignments[d]
            indices = []
            for lbl in device_labels:
                start = idx_ptr[lbl]
                end = start + samples_per_class
                indices.extend(label_to_indices[lbl][start:end])
                idx_ptr[lbl] += samples_per_class

            np.random.shuffle(indices)
            cluster_device_data[c][d] = x[indices]
            cluster_device_labels[c][d] = y[indices]

    return cluster_device_data, cluster_device_labels

import numpy as np
from collections import defaultdict
from torchvision import datasets, transforms

import numpy as np
from collections import defaultdict

def dirichlet_partition_full_mnist(x, y, num_clusters=3, devices_per_cluster=10, alpha=0.5, seed=42):
    np.random.seed(seed)
    total_devices = devices_per_cluster
    labels = np.unique(y)

    # Output: each cluster has its own split
    cluster_device_data = {}
    cluster_device_labels = {}

    for c in range(num_clusters):
        # Each cluster gets access to the full dataset
        class_indices = {label: np.where(y == label)[0] for label in labels}
        for indices in class_indices.values():
            np.random.shuffle(indices)

        device_indices = defaultdict(list)

        # For each class, partition among devices using Dirichlet
        for label in labels:
            indices = class_indices[label]
            proportions = np.random.dirichlet([alpha] * total_devices)
            proportions = (proportions * len(indices)).astype(int)
            proportions[-1] += len(indices) - proportions.sum()  # fix rounding

            start = 0
            for device_id, count in enumerate(proportions):
                end = start + count
                device_indices[device_id].extend(indices[start:end])
                start = end

        # Store data for each device in this cluster
        cluster_device_data[c] = {}
        cluster_device_labels[c] = {}
        for d in range(devices_per_cluster):
            dev_idx = device_indices[d]
            np.random.shuffle(dev_idx)
            cluster_device_data[c][d] = x[dev_idx]
            cluster_device_labels[c][d] = y[dev_idx]

    return cluster_device_data, cluster_device_labels
