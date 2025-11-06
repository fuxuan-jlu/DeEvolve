import csv
import random
from datasets import load_dataset

def load_delivery_points(start_index, end_index, dataset_name="Cainiao-AI/LaDe-D", split="delivery_yt"):
    ds = load_dataset(dataset_name)
    train_dataset = ds[split]
    #print(train_dataset)
    if isinstance(train_dataset, list):
        delivery_points = [(point['lat'], point['lng']) for point in train_dataset[start_index:end_index]]
    else:
        delivery_points = [(point['lat'], point['lng']) for point in list(train_dataset)[start_index:end_index]]

    return delivery_points

def save_random_points_to_csv(start_index, end_index,n_points, dataset_name="Cainiao-AI/LaDe-D", split="delivery_yt",
                              filename="dataset.csv"):
    # Load the delivery points
    delivery_points = load_delivery_points(start_index, end_index,dataset_name, split)

    # Randomly select n_points
    selected_points = random.sample(delivery_points, n_points)

    # Write to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['latitude', 'longitude'])
        # Write points
        for point in selected_points:
            writer.writerow([point[0], point[1]])

    print(f"Successfully saved {n_points} random points to {filename}")


def load_points_from_csv(filename="dataset.csv"):

    delivery_points = []

    with open(filename, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                lat = float(row['latitude'])
                lng = float(row['longitude'])
                delivery_points.append((lat, lng))
            except (ValueError, KeyError) as e:
                print(f"Skipping invalid row: {row}. Error: {e}")

    return delivery_points

def main():
    # Parameters
    n_points = 101

    # Call the function to save random points
    save_random_points_to_csv(
        2500,
        3001,
        n_points=n_points,
    )
    points = load_points_from_csv()
    print(points)

if __name__ == "__main__":
    main()
# Example usage:
# save_random_points_to_csv(100, 0, 1000)  # Select 100 random points from first 1000 points