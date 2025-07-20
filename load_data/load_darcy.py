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
    indices_x = torch.linspace(0, tensor.shape[-2] - 1, target_resolution).long()
    indices_y = torch.linspace(0, tensor.shape[-1] - 1, target_resolution).long()
    return tensor[:, indices_x][
        :, :, indices_y
    ]  # Downsample equally in both spatial dimensions


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
    x_dataset = file["nu"][:1000]
    y_dataset = file["tensor"][:1000]

    n_samples = x_dataset.shape[0]
    n_train = int(0.9 * n_samples)

    x_train_list, y_train_list = [], []
    x_test_list, y_test_list = [], []

    # Process training data in chunks
    for i in range(0, n_train, batch_size):
        x_batch = torch.tensor(x_dataset[i : i + batch_size, ...], dtype=torch.float32)
        y_batch = torch.tensor(
            y_dataset[i : i + batch_size, ...], dtype=torch.float32
        ).squeeze(1)

        # Downsample space dimension by selecting points
        x_batch = downsample_select(x_batch, resolution)
        y_batch = downsample_select(y_batch, resolution)

        x_train_list.append(x_batch)
        y_train_list.append(y_batch)

    # Process testing data in chunks
    for i in range(n_train, n_samples, batch_size):
        x_batch = torch.tensor(x_dataset[i : i + batch_size, ...], dtype=torch.float32)
        y_batch = torch.tensor(
            y_dataset[i : i + batch_size, ...], dtype=torch.float32
        ).squeeze(1)

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
    default="darcy",
    choices=["darcy"],
    help="Name of the dataset to process and save",
)
parser.add_argument(
    "--resolution_start",
    type=int,
    default=4,
    help="Start of the range of resolutions to save (inclusive)",
)
parser.add_argument(
    "--resolution_end",
    type=int,
    default=7,
    help="End of the range of resolutions to save (inclusive)",
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
    default="data",
)
args = parser.parse_args()
args.resolutions = [2**i for i in range(args.resolution_start, args.resolution_end + 1)]
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
