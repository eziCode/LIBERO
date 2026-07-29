import init_path
import argparse
import os
import time

import libero.libero.utils.download_utils as download_utils
from libero.libero import get_libero_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-dir",
        type=str,
        default=get_libero_path("datasets"),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        choices=["all", "libero_goal", "libero_spatial", "libero_object", "libero_100", "libero_90", "libero_10"],
        default="all",
    )
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Use Hugging Face instead of original download links"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of tasks to download"
    )
    parser.add_argument(
        "--random-tasks",
        type=int,
        default=0,
        help="Number of random tasks to download from the specified dataset"
    )
    return parser.parse_args()


def main():

    args = parse_args()

    # Ask users to specify the download directory of datasets
    os.makedirs(args.download_dir, exist_ok=True)
    print(f"Datasets downloaded to {args.download_dir}")
    print(f"Downloading {args.datasets} datasets")

    if args.use_huggingface:
        print("Using Hugging Face as the download source")
    else:
        print("Using original download links (note: these may expire soon)")
        input_str = input("Download from original links may lead to failures. Do you want to continue? (y/n): ")
        if input_str.lower() != 'y':
            print("Switching to Hugging Face as the download source...")
            args.use_huggingface = True

    task_names = None
    if args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",")]
    
    if args.random_tasks > 0:
        if args.datasets == "all":
            print("Error: --random-tasks requires a specific dataset (e.g., libero_90)")
            return
        
        from libero.libero.benchmark.libero_suite_task_map import libero_task_map
        import random
        
        all_tasks = libero_task_map.get(args.datasets, [])
        if not all_tasks:
            print(f"Error: No tasks found for dataset {args.datasets}")
            return
        
        # Filter out already downloaded tasks
        dataset_dir = os.path.join(args.download_dir, args.datasets)
        existing_tasks = []
        if os.path.exists(dataset_dir):
            for f in os.listdir(dataset_dir):
                if f.endswith("_demo.hdf5"):
                    existing_tasks.append(f.replace("_demo.hdf5", ""))
        
        available_tasks = [t for t in all_tasks if t not in existing_tasks]
        
        if not available_tasks:
            print(f"All tasks for {args.datasets} are already downloaded.")
            return
            
        num_to_pick = min(args.random_tasks, len(available_tasks))
        task_names = random.sample(available_tasks, num_to_pick)
        print(f"Randomly selected {num_to_pick} tasks: {task_names}")

    # If not, download
    download_utils.libero_dataset_download(
        download_dir=args.download_dir, 
        datasets=args.datasets,
        use_huggingface=args.use_huggingface,
        task_names=task_names
    )


    # wait for 1 second
    time.sleep(1)
    print("\n\n\n")

    # Check if datasets exist first
    download_utils.check_libero_dataset(download_dir=args.download_dir)


if __name__ == "__main__":
    main()
