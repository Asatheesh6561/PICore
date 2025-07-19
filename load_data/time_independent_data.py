import torch
import h5py


def downsample_select(tensor, target_resolution):
    """
    Downsample the spatial dimension by selecting evenly spaced points.

    Args:
        tensor (torch.Tensor): Shape (batch, time, space)
        target_resolution (int): Target spatial resolution (must be power of 2)

    Returns:
        torch.Tensor: Downsampled tensor with shape (batch, time, target_resolution)
    """
    dim = len(tensor.shape)
    indices = torch.linspace(0, tensor.shape[-1] - 1, target_resolution).long()
    time_dim = tensor.shape[1]
    return tensor[:, :20][:, :, indices]  # Select along time and spatial dimensions


def process_and_save(
    file_path, save_path_train, save_path_test, resolution=256, batch_size=100
):
    """
    Load, downsample, and save dataset in batches to prevent OOM errors.

    Args:
        file_path (str): Path to the HDF5 file.
        save_path_train (str): Path to save training data.
        save_path_test (str): Path to save test data.
        resolution (int): Target spatial resolution (must be power of 2).
        batch_size (int): Number of samples per batch.
    """
    assert (
        resolution > 0 and (resolution & (resolution - 1)) == 0
    ), "Resolution must be a power of 2!"
    assert resolution <= 1024, "Resolution must be ≤ 1024 (original size)!"

    print(f"Using resolution: {resolution}")

    # Load dataset
    file = h5py.File(file_path, "r")
    dataset = file["tensor"][:]
    time_dim = dataset.shape[1]

    n_samples = dataset.shape[0]
    n_train = int(0.9 * n_samples)

    x_train_list, y_train_list = [], []
    x_test_list, y_test_list = [], []

    # Process training data in chunks
    for i in range(0, n_train, batch_size):
        x_batch = torch.tensor(
            dataset[i : i + batch_size, 0:1, :], dtype=torch.float32
        ).repeat(1, time_dim, 1)
        y_batch = torch.tensor(dataset[i : i + batch_size, :, :], dtype=torch.float32)

        # Downsample space dimension by selecting points
        x_batch = downsample_select(x_batch, resolution)
        y_batch = downsample_select(y_batch, resolution)

        x_train_list.append(x_batch)
        y_train_list.append(y_batch)

    # Process testing data in chunks
    for i in range(n_train, n_samples, batch_size):
        x_batch = torch.tensor(
            dataset[i : i + batch_size, 0:1, :], dtype=torch.float32
        ).repeat(1, time_dim, 1)
        y_batch = torch.tensor(dataset[i : i + batch_size, :, :], dtype=torch.float32)

        # Downsample space dimension by selecting points
        x_batch = downsample_select(x_batch, resolution)
        y_batch = downsample_select(y_batch, resolution)

        x_test_list.append(x_batch)
        y_test_list.append(y_batch)

    # Concatenate batches
    x_train, y_train = torch.cat(x_train_list, dim=0), torch.cat(y_train_list, dim=0)
    x_test, y_test = torch.cat(x_test_list, dim=0), torch.cat(y_test_list, dim=0)

    # Save to disk
    torch.save({"x": x_train, "y": y_train}, save_path_train)
    torch.save({"x": x_test, "y": y_test}, save_path_test)

    print(f"Data saved successfully at resolution {resolution}!")


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    type=str,
    default="advection",
    choices=["advection", "burgers"],
    help="Name of the dataset to process and save",
)
parser.add_argument(
    "--resolutions",
    type=int,
    nargs="+",
    default=[2**i for i in range(4, 11)],
    help="List of resolutions to save",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=100,
    help="Batch size for processing and saving",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="/cmlscratch/anirudhs/DENO/data/",
)
args = parser.parse_args()

for resolution in args.resolutions:
    file_path = f"{args.data_dir}/original_data/{args.dataset_name}.hdf5"
    save_path_train = (
        f"{args.data_dir}/{args.dataset_name}/{args.dataset_name}_train_{resolution}.pt"
    )
    save_path_test = (
        f"{args.data_dir}/{args.dataset_name}/{args.dataset_name}_test_{resolution}.pt"
    )

    process_and_save(
        file_path,
        save_path_train,
        save_path_test,
        resolution=resolution,
        batch_size=args.batch_size,
    )
