import argparse
import time
import rerun as rr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rrd_file")
    args = parser.parse_args()

    server = rr.server.Server(datasets={"datasets": [args.rrd_file]})
    recording = server.client().get_dataset("datasets")

    query = recording.reader(index="time")

    df = query.to_pandas()

    print("Available columns in your RRD file:")
    for col in df.columns:
        print(f" - {col}")
    print("-" * 40)

    column_name = "/lidar/points:Points3D:positions"
    
    for _, row in df.iterrows():
        xyz = row[column_name]

        if xyz is not None and len(xyz) > 0:
            print(xyz[:3])
            time.sleep(1)

if __name__ == "__main__":
    main()