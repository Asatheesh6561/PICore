import torch
import h5py
import psutil
import os
import gc
import argparse
from pathlib import Path


def downsample_select(tensor, target_resolution):
    """
    Downsample the spatial dimension by selecting evenly spaced points.

    Args:
        tensor (torch.Tensor): Shape (batch, time, space1, space2, ...)
        target_resolution (int): Target spatial resolution (must be power of 2)

    Returns:
        torch.Tensor: Downsampled tensor with shape (batch, time, target_resolution, target_resolution, ...)
    """
    spatial_dims = tensor.shape[2:]  # Extract spatial dimensions
    indices = [torch.linspace(0, s - 1, target_resolution).long() for s in spatial_dims]
    return tensor[
        (slice(None), slice(None)) + tuple(torch.meshgrid(*indices, indexing="ij"))
    ]


def process_in_chunks(data, start_idx, end_idx, resolution, batch_size):
    """Process data in chunks to prevent OOM errors."""
    processed = []
    for i in range(start_idx, end_idx, batch_size):
        batch = data[i : min(i + batch_size, end_idx)]
        batch = downsample_select(batch, resolution)
        processed.append(batch)
        del batch
        gc.collect()
        torch.cuda.empty_cache()
    return torch.cat(processed, dim=0)


def process_and_save(
    file_path, save_path_train, save_path_test, resolution=256, batch_size=1
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
    # Validate resolution
    if not (resolution > 0 and (resolution & (resolution - 1)) == 0):
        raise ValueError("Resolution must be a power of 2!")
    if resolution > 1024:
        raise ValueError("Resolution must be ≤ 1024 (original size)!")

    print(f"Processing with resolution: {resolution}")

    # Load dataset
    process = psutil.Process(os.getpid())
    print(f"Memory before loading: {process.memory_info().rss / (1024**2):.2f} MB")

    try:
        with h5py.File(file_path, "r") as file:
            input = file["density"][:1000]
            x_data = torch.tensor(input[:, 0:1, ...]).repeat(1, input.shape[1], 1, 1)
            y_data = torch.tensor(input)
            del input
            gc.collect()
            print(
                f"Memory after loading: {process.memory_info().rss / (1024**2):.2f} MB"
            )

            n_samples = x_data.shape[0]
            n_train = int(0.9 * n_samples)

            # Process training data
            x_train = process_in_chunks(
                x_data[:n_train], 0, n_train, resolution, batch_size
            )
            y_train = process_in_chunks(
                y_data[:n_train], 0, n_train, resolution, batch_size
            )

            # Process test data
            x_test = process_in_chunks(
                x_data[n_train:], 0, n_samples - n_train, resolution, batch_size
            )
            y_test = process_in_chunks(
                y_data[n_train:], 0, n_samples - n_train, resolution, batch_size
            )

            # Ensure directories exist
            Path(save_path_train).parent.mkdir(parents=True, exist_ok=True)
            Path(save_path_test).parent.mkdir(parents=True, exist_ok=True)

            # Save to disk
            torch.save({"x": x_train, "y": y_train}, save_path_train)
            torch.save({"x": x_test, "y": y_test}, save_path_test)
            print(f"Successfully saved data at resolution {resolution}")

    except Exception as e:
        print(f"Error processing data: {str(e)}")
    finally:
        # Clean up
        if "x_data" in locals():
            del x_data, y_data, x_train, y_train, x_test, y_test
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Process and save downsampled dataset")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ns_com",
        choices=["ns_com"],
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
        default=10,
        help="End of the range of resolutions to save (inclusive)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for processing and saving",
    )
    parser.add_argument(
        "--dimension", type=int, default=2, help="Dimension of the data (1D, 2D, etc.)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/cmlscratch/anirudhs/DENO/data",
        help="Base directory for data",
    )

    args = parser.parse_args()
    resolutions = [2**i for i in range(args.resolution_start, args.resolution_end + 1)]

    base_path = Path(args.data_dir)
    dataset_dir = base_path / f"{args.dataset_name}_{args.dimension}d"
    dataset_dir.mkdir(exist_ok=True)

    input_file = (
        base_path / "original_data" / f"{args.dataset_name}_{args.dimension}d.hdf5"
    )

    for resolution in resolutions:
        output_train = dataset_dir / f"{args.dataset_name}_train_{resolution}.pt"
        output_test = dataset_dir / f"{args.dataset_name}_test_{resolution}.pt"

        process_and_save(
            str(input_file),
            str(output_train),
            str(output_test),
            resolution=resolution,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
